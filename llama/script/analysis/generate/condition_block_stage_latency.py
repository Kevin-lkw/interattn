import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .methods.condition_block_triton_impl.core import (
    _condition_block_decode_output_fused_triton,
    _run_condition_block_selection_stats,
    _select_prompt_blocks_triton,
)
from .methods.condition_block_triton_impl.legacy import (
    _condition_block_stage2_reduce_kernel,
)
from ..runner_utils import set_seed


@triton.jit(
    do_not_specialize=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
    do_not_specialize_on_alignment=[
        "n_blocks",
        "suffix_len",
        "n_chunks",
        "k_suffix_head_stride",
        "k_suffix_token_stride",
        "v_suffix_head_stride",
        "v_suffix_token_stride",
    ],
)
def _dummy_selected_sparse_attention_kernel(
    q_ptr,
    k_block_ptr,
    v_block_ptr,
    k_bar_ptr,
    v_bar_ptr,
    counts_ptr,
    selected_ptr,
    k_suffix_ptr,
    v_suffix_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    n_blocks,
    suffix_len,
    k_suffix_head_stride,
    k_suffix_token_stride,
    v_suffix_head_stride,
    v_suffix_token_stride,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    n_chunks,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    kv_head = tl.program_id(0)
    chunk = tl.program_id(1)
    m_off = tl.arange(0, BLOCK_M)
    n_off = tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    m_mask = m_off < group_size
    block_idx = chunk * BLOCK_N + n_off
    block_mask = block_idx < n_blocks
    d_mask = d_off < head_dim
    row = kv_head * group_size + m_off

    q = tl.load(
        q_ptr + row[:, None] * head_dim + d_off[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    count = tl.load(counts_ptr + block_idx, mask=block_mask, other=0)
    active_block = block_mask & (count > 0)
    selected = tl.load(
        selected_ptr + kv_head * n_blocks + block_idx,
        mask=block_mask,
        other=0,
    ).to(tl.int1) & active_block

    softmax_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    softmax_l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Unselected blocks: one representative key/value per block.
    active_rep = active_block & (~selected)
    stat_off = ((kv_head * n_blocks + block_idx[:, None]) * head_dim) + d_off[None, :]
    k_rep = tl.load(
        k_bar_ptr + stat_off,
        mask=active_rep[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    rep_scores = tl.dot(q, tl.trans(k_rep), out_dtype=tl.float32) * scale
    rep_scores = rep_scores + tl.log(tl.maximum(count, 1).to(tl.float32))[None, :]
    rep_scores = tl.where(
        m_mask[:, None] & active_rep[None, :], rep_scores, -float("inf")
    )
    has_rep = tl.sum(active_rep.to(tl.int32), axis=0) > 0
    rep_m = tl.max(rep_scores, axis=1)
    new_m = tl.where(has_rep & m_mask, rep_m, softmax_m)
    alpha = tl.where(has_rep & m_mask, tl.exp(softmax_m - new_m), 1.0)
    beta = tl.where(
        m_mask[:, None] & active_rep[None, :],
        tl.exp(rep_scores - new_m[:, None]),
        0.0,
    )
    v_rep = tl.load(
        v_bar_ptr + stat_off,
        mask=active_rep[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    acc = acc * alpha[:, None] + tl.dot(
        beta.to(tl.bfloat16), v_rep, out_dtype=tl.float32
    )
    softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
    softmax_m = new_m

    # Selected blocks: exact token attention, using the same 16-token MMA tile
    # as the production kernel. PAGE_SIZE=32 therefore consumes two tiles.
    for local_block in tl.static_range(0, BLOCK_N):
        page_selected = tl.sum(
            (selected & (n_off == local_block)).to(tl.int32), axis=0
        ) > 0
        if page_selected:
            page = chunk * BLOCK_N + local_block
            page_count = tl.load(counts_ptr + page)
            for page_start in tl.static_range(0, PAGE_SIZE, BLOCK_N):
                token_idx = page_start + n_off
                token_active = token_idx < page_count
                token_off = (
                    ((kv_head * n_blocks + page) * PAGE_SIZE + token_idx[:, None])
                    * head_dim
                    + d_off[None, :]
                )
                k = tl.load(
                    k_block_ptr + token_off,
                    mask=token_active[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.bfloat16)
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
                scores = tl.where(
                    m_mask[:, None] & token_active[None, :],
                    scores,
                    -float("inf"),
                )
                tile_m = tl.max(scores, axis=1)
                new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
                alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
                beta = tl.where(
                    m_mask[:, None] & token_active[None, :],
                    tl.exp(scores - new_m[:, None]),
                    0.0,
                )
                v = tl.load(
                    v_block_ptr + token_off,
                    mask=token_active[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.bfloat16)
                acc = acc * alpha[:, None] + tl.dot(
                    beta.to(tl.bfloat16), v, out_dtype=tl.float32
                )
                softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
                softmax_m = new_m

    if chunk == n_chunks - 1:
        suffix_start = 0
        while suffix_start < suffix_len:
            suffix_idx = suffix_start + n_off
            suffix_active = suffix_idx < suffix_len
            k_suffix_off = (
                kv_head * k_suffix_head_stride
                + suffix_idx[:, None] * k_suffix_token_stride
                + d_off[None, :]
            )
            k = tl.load(
                k_suffix_ptr + k_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                m_mask[:, None] & suffix_active[None, :], scores, -float("inf")
            )
            tile_m = tl.max(scores, axis=1)
            new_m = tl.where(m_mask, tl.maximum(softmax_m, tile_m), softmax_m)
            alpha = tl.where(m_mask, tl.exp(softmax_m - new_m), 1.0)
            beta = tl.where(
                m_mask[:, None] & suffix_active[None, :],
                tl.exp(scores - new_m[:, None]),
                0.0,
            )
            v_suffix_off = (
                kv_head * v_suffix_head_stride
                + suffix_idx[:, None] * v_suffix_token_stride
                + d_off[None, :]
            )
            v = tl.load(
                v_suffix_ptr + v_suffix_off,
                mask=suffix_active[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                beta.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            softmax_l = softmax_l * alpha + tl.sum(beta, axis=1)
            softmax_m = new_m
            suffix_start += BLOCK_N

    partial_off = row * n_chunks + chunk
    tl.store(partial_m_ptr + partial_off, softmax_m, mask=m_mask)
    tl.store(partial_l_ptr + partial_off, softmax_l, mask=m_mask)
    tl.store(
        partial_acc_ptr + partial_off[:, None] * head_dim + d_off[None, :],
        acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark condition-block Triton stage latency. "
            "This uses synthetic tensors and does not load a model."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--summary-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help=(
            "Dtype of the per-block summaries (k_bar/k_max/k_min/v_bar) that are "
            "re-read from HBM at every decode step. The kernels convert to FP32 in "
            "registers, so bfloat16 halves summary IO without touching kernel code."
        ),
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--suffix-tokens", type=int, default=0)
    parser.add_argument("--selected-ratios", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--include-full-sdpa", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/condition_block_stage_latency.jsonl"))
    return parser.parse_args()


def cuda_time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def build_synthetic_prefix(
    *, n_kv_heads, context_tokens, block_size, head_dim, dtype, device, summary_dtype=torch.float32
):
    n_blocks = math.ceil(context_tokens / block_size)
    total_tokens = n_blocks * block_size
    k_block = torch.randn(
        (n_kv_heads, n_blocks, block_size, head_dim),
        device=device,
        dtype=dtype,
    )
    v_block = torch.randn_like(k_block)
    if total_tokens != context_tokens:
        valid_tokens = torch.arange(total_tokens, device=device).reshape(n_blocks, block_size) < context_tokens
        k_block = k_block.masked_fill(~valid_tokens.view(1, n_blocks, block_size, 1), 0.0)
        v_block = v_block.masked_fill(~valid_tokens.view(1, n_blocks, block_size, 1), 0.0)
    else:
        valid_tokens = torch.ones((n_blocks, block_size), device=device, dtype=torch.bool)
    counts = valid_tokens.sum(dim=1).long()
    valid = valid_tokens.view(1, n_blocks, block_size, 1)
    size_float = counts.clamp_min(1).float()
    k_bar = (k_block * valid).sum(dim=2, dtype=torch.float32) / size_float.view(1, n_blocks, 1)
    v_bar = (v_block * valid).sum(dim=2, dtype=torch.float32) / size_float.view(1, n_blocks, 1)
    k_for_max = k_block.masked_fill(~valid, float("-inf"))
    k_for_min = k_block.masked_fill(~valid, float("inf"))
    v_norm = torch.linalg.vector_norm(v_block, dim=-1, dtype=torch.float32)
    v_norm = v_norm.masked_fill(~valid_tokens.view(1, n_blocks, block_size), float("-inf"))
    return {
        "k_block_attn": k_block,
        "v_block_attn": v_block,
        "k_bar": k_bar.to(summary_dtype),
        "v_bar": v_bar.to(summary_dtype),
        "k_max": k_for_max.amax(dim=2).float().to(summary_dtype),
        "k_min": k_for_min.amin(dim=2).float().to(summary_dtype),
        "v_norm_max": v_norm.amax(dim=2),
        "v_norm_all": v_norm.amax(dim=2).amax(dim=-1),
        "block_valid_counts": counts,
    }


def build_dummy_selected(n_kv_heads, n_blocks, ratio, device):
    n_selected = int(round(float(ratio) * n_blocks))
    selected = torch.zeros((n_kv_heads, n_blocks), device=device, dtype=torch.bool)
    if n_selected <= 0:
        return selected
    # Evenly spaced pages avoid accidentally benchmarking only the first chunks.
    idx = torch.linspace(0, n_blocks - 1, n_selected, device=device).round().long().unique()
    selected[:, idx] = True
    return selected


def run_sparse_attention_dummy(
    *,
    q,
    prefix,
    selected,
    k_suffix,
    v_suffix,
    group_size,
    head_dim,
    block_size,
    output_dtype,
):
    n_kv_heads = int(prefix["k_block_attn"].shape[0])
    n_blocks = int(prefix["block_valid_counts"].numel())
    block_n = 16
    n_chunks = triton.cdiv(n_blocks, block_n)
    rows = n_kv_heads * group_size
    partial_acc = torch.empty((rows, n_chunks, head_dim), device=q.device, dtype=torch.float32)
    partial_m = torch.empty((rows, n_chunks), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((rows, n_chunks), device=q.device, dtype=torch.float32)
    out = torch.empty((rows, head_dim), device=q.device, dtype=output_dtype)
    _dummy_selected_sparse_attention_kernel[(n_kv_heads, n_chunks)](
        q,
        prefix["k_block_attn"].contiguous(),
        prefix["v_block_attn"].contiguous(),
        prefix["k_bar"].contiguous(),
        prefix["v_bar"].contiguous(),
        prefix["block_valid_counts"].contiguous(),
        selected.contiguous(),
        k_suffix,
        v_suffix,
        partial_acc,
        partial_m,
        partial_l,
        n_blocks,
        int(k_suffix.shape[1]),
        k_suffix.stride(0),
        k_suffix.stride(1),
        v_suffix.stride(0),
        v_suffix.stride(1),
        group_size,
        head_dim,
        n_chunks,
        head_dim**-0.5,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        PAGE_SIZE=int(block_size),
        num_warps=4,
    )
    _condition_block_stage2_reduce_kernel[(rows,)](
        partial_acc,
        partial_m,
        partial_l,
        out,
        n_chunks,
        head_dim,
        BLOCK_C=triton.next_power_of_2(n_chunks),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out


def run_full_sdpa(q_grouped, prefix):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    context_tokens = int(prefix["block_valid_counts"].sum().item())
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
    k = prefix["k_block_attn"].reshape(n_kv_heads, -1, head_dim)[:, :context_tokens]
    v = prefix["v_block_attn"].reshape(n_kv_heads, -1, head_dim)[:, :context_tokens]
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=head_dim**-0.5,
        enable_gqa=True,
    )


def main():
    args = parse_args()
    set_seed(42)
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This benchmark requires a CUDA device.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads.")
    group_size = args.q_heads // args.kv_heads
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary_dtype = torch.bfloat16 if args.summary_dtype == "bfloat16" else torch.float32

    with args.output.open("w", encoding="utf-8") as handle:
        for context_tokens in args.contexts:
            prefix = build_synthetic_prefix(
                n_kv_heads=args.kv_heads,
                context_tokens=int(context_tokens),
                block_size=args.block_size,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
                summary_dtype=summary_dtype,
            )
            q_grouped = torch.randn(
                (args.kv_heads, group_size, 1, args.head_dim),
                device=device,
                dtype=dtype,
            )
            q = q_grouped.reshape(args.kv_heads * group_size, args.head_dim).contiguous()
            k_suffix = torch.randn(
                (args.kv_heads, int(args.suffix_tokens), args.head_dim),
                device=device,
                dtype=dtype,
            )
            v_suffix = torch.randn_like(k_suffix)

            workspace = {}
            row = {
                "context_tokens": int(context_tokens),
                "block_size": int(args.block_size),
                "summary_dtype": args.summary_dtype,
                "n_blocks": int(prefix["block_valid_counts"].numel()),
                "suffix_tokens": int(args.suffix_tokens),
                "stage": "selection_stats_reduce",
                "latency_ms": cuda_time_ms(
                    lambda: _run_condition_block_selection_stats(q_grouped, prefix, workspace),
                    args.warmup,
                    args.iters,
                ),
            }
            print(json.dumps(row), flush=True)
            handle.write(json.dumps(row) + "\n")

            row = {
                "context_tokens": int(context_tokens),
                "block_size": int(args.block_size),
                "summary_dtype": args.summary_dtype,
                "n_blocks": int(prefix["block_valid_counts"].numel()),
                "suffix_tokens": int(args.suffix_tokens),
                "stage": "selection_materialize",
                "latency_ms": cuda_time_ms(
                    lambda: _select_prompt_blocks_triton(q_grouped, prefix, args.eps),
                    args.warmup,
                    args.iters,
                ),
            }
            print(json.dumps(row), flush=True)
            handle.write(json.dumps(row) + "\n")

            selected_actual, _, _, _, _ = _select_prompt_blocks_triton(q_grouped, prefix, args.eps)
            torch.cuda.synchronize()
            actual_selected_ratio = float(selected_actual[0, 0, 0].float().mean().item())
            fused_workspace = {}
            suffix_len_dev = torch.full(
                (), int(args.suffix_tokens), dtype=torch.int32, device=device
            )
            row = {
                "context_tokens": int(context_tokens),
                "block_size": int(args.block_size),
                "summary_dtype": args.summary_dtype,
                "n_blocks": int(prefix["block_valid_counts"].numel()),
                "suffix_tokens": int(args.suffix_tokens),
                "stage": "production_fused_selection_attention",
                "eps": float(args.eps),
                "actual_selected_ratio": actual_selected_ratio,
                "latency_ms": cuda_time_ms(
                    lambda: _condition_block_decode_output_fused_triton(
                        q_grouped=q_grouped,
                        prompt_prefix=prefix,
                        k_suffix=k_suffix,
                        v_suffix=v_suffix,
                        suffix_len_dev=suffix_len_dev,
                        eps=args.eps,
                        page_size=args.block_size,
                        store_selected=False,
                        output_dtype=dtype,
                        workspace=fused_workspace,
                    ),
                    args.warmup,
                    args.iters,
                ),
            }
            print(json.dumps(row), flush=True)
            handle.write(json.dumps(row) + "\n")

            if args.include_full_sdpa:
                row = {
                    "context_tokens": int(context_tokens),
                    "block_size": int(args.block_size),
                    "n_blocks": int(prefix["block_valid_counts"].numel()),
                    "suffix_tokens": int(args.suffix_tokens),
                    "stage": "full_sdpa_decode_attention",
                    "latency_ms": cuda_time_ms(
                        lambda: run_full_sdpa(q_grouped, prefix),
                        args.warmup,
                        args.iters,
                    ),
                }
                print(json.dumps(row), flush=True)
                handle.write(json.dumps(row) + "\n")

            for ratio in args.selected_ratios:
                selected = build_dummy_selected(
                    args.kv_heads,
                    int(prefix["block_valid_counts"].numel()),
                    ratio,
                    device,
                )
                effective_prompt_candidates = (
                    selected.sum(dim=1).float().mean().item() * args.block_size
                    + (int(prefix["block_valid_counts"].numel()) - selected.sum(dim=1).float().mean().item())
                )
                full_prompt_candidates = float(context_tokens)
                theoretical_fraction = (
                    effective_prompt_candidates + int(args.suffix_tokens)
                ) / max(full_prompt_candidates + int(args.suffix_tokens), 1.0)
                row = {
                    "context_tokens": int(context_tokens),
                    "block_size": int(args.block_size),
                    "n_blocks": int(prefix["block_valid_counts"].numel()),
                    "suffix_tokens": int(args.suffix_tokens),
                    "stage": "dummy_sparse_attention",
                    "selected_ratio": float(ratio),
                    "selected_blocks": int(selected[0].sum().item()),
                    "theoretical_candidate_fraction": theoretical_fraction,
                    "latency_ms": cuda_time_ms(
                        lambda selected=selected: run_sparse_attention_dummy(
                            q=q,
                            prefix=prefix,
                            selected=selected,
                            k_suffix=k_suffix,
                            v_suffix=v_suffix,
                            group_size=group_size,
                            head_dim=args.head_dim,
                            block_size=args.block_size,
                            output_dtype=dtype,
                        ),
                        args.warmup,
                        args.iters,
                    ),
                }
                print(json.dumps(row), flush=True)
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
