import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from ..condition_block_stage_latency import (
    build_dummy_selected,
    build_synthetic_prefix,
    cuda_time_ms,
    run_sparse_attention_dummy,
)
from ...runner_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic decode-attention benchmark for FlexAttention experiments. "
            "This compares full SDPA, the existing Triton dummy hybrid attention, "
            "FlexAttention token block sparsity, and FlexAttention over compact "
            "representative/selected-token K/V."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--condition-block-size", type=int, default=32)
    parser.add_argument(
        "--flex-block-size",
        type=int,
        default=128,
        help="BlockMask granularity. In this PyTorch build, 128 is the working decode block size.",
    )
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--selected-ratios", nargs="+", type=float, default=[0.0, 0.1, 0.25])
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode for FlexAttention. max-autotune is slower to compile "
            "but can find valid FlexDecoding kernels for 32-token block masks."
        ),
    )
    parser.add_argument(
        "--no-compile-dynamic",
        action="store_true",
        help="Disable dynamic=True in torch.compile. Dynamic mode is recommended when sweeping KV lengths.",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run small correctness checks and exit.",
    )
    parser.add_argument("--sanity-context", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("/tmp/condition_block_flex_stage_latency.jsonl"))
    return parser.parse_args()


def wall_time_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def flatten_prompt_kv(prefix, context_tokens):
    k_prompt = prefix["k_block_attn"].reshape(prefix["k_block_attn"].shape[0], -1, prefix["k_block_attn"].shape[-1])
    v_prompt = prefix["v_block_attn"].reshape(prefix["v_block_attn"].shape[0], -1, prefix["v_block_attn"].shape[-1])
    return k_prompt[:, :context_tokens], v_prompt[:, :context_tokens]


def run_full_sdpa_with_suffix(q_grouped, prefix, k_suffix, v_suffix):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    context_tokens = int(prefix["block_valid_counts"].sum().item())
    k_prompt, v_prompt = flatten_prompt_kv(prefix, context_tokens)
    k = torch.cat([k_prompt, k_suffix], dim=1).unsqueeze(0)
    v = torch.cat([v_prompt, v_suffix], dim=1).unsqueeze(0)
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
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


def make_flex_dense_call(compiled_flex, q_grouped, prefix, k_suffix, v_suffix):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    context_tokens = int(prefix["block_valid_counts"].sum().item())
    k_prompt, v_prompt = flatten_prompt_kv(prefix, context_tokens)
    k = torch.cat([k_prompt, k_suffix], dim=1).unsqueeze(0)
    v = torch.cat([v_prompt, v_suffix], dim=1).unsqueeze(0)
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)

    def call():
        return compiled_flex(
            q,
            k,
            v,
            scale=head_dim**-0.5,
            enable_gqa=True,
        )

    return call


def build_flex_block_selected(*, context_tokens, suffix_tokens, flex_block_size, ratio, device):
    n_prompt_blocks = math.ceil(context_tokens / flex_block_size)
    n_total_blocks = math.ceil((context_tokens + suffix_tokens) / flex_block_size)
    selected = torch.zeros((n_total_blocks,), device=device, dtype=torch.bool)
    n_selected = int(round(float(ratio) * n_prompt_blocks))
    if n_selected > 0:
        idx = torch.linspace(0, n_prompt_blocks - 1, n_selected, device=device).round().long().unique()
        selected[idx] = True
    if suffix_tokens > 0:
        suffix_first_block = context_tokens // flex_block_size
        selected[suffix_first_block:] = True
    return selected


def make_flex_block_call(compiled_flex, q_grouped, prefix, k_suffix, v_suffix, flex_block_size, ratio):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    context_tokens = int(prefix["block_valid_counts"].sum().item())
    k_prompt, v_prompt = flatten_prompt_kv(prefix, context_tokens)
    k = torch.cat([k_prompt, k_suffix], dim=1).unsqueeze(0)
    v = torch.cat([v_prompt, v_suffix], dim=1).unsqueeze(0)
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
    selected_blocks = build_flex_block_selected(
        context_tokens=context_tokens,
        suffix_tokens=int(k_suffix.shape[1]),
        flex_block_size=flex_block_size,
        ratio=ratio,
        device=q.device,
    )

    def mask_mod(_b, _h, _q_idx, kv_idx):
        return selected_blocks[kv_idx // flex_block_size]

    block_mask, build_ms = wall_time_ms(
        lambda: create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=1,
            KV_LEN=int(k.shape[2]),
            device=q.device,
            BLOCK_SIZE=flex_block_size,
        )
    )

    def call():
        return compiled_flex(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
        )

    return call, build_ms, int(selected_blocks[: math.ceil(context_tokens / flex_block_size)].sum().item())


def make_flex_block_from_kv_blocks_call(
    compiled_flex,
    q_grouped,
    prefix,
    k_suffix,
    v_suffix,
    flex_block_size,
    ratio,
):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    context_tokens = int(prefix["block_valid_counts"].sum().item())
    k_prompt, v_prompt = flatten_prompt_kv(prefix, context_tokens)
    k = torch.cat([k_prompt, k_suffix], dim=1).unsqueeze(0)
    v = torch.cat([v_prompt, v_suffix], dim=1).unsqueeze(0)
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
    selected_blocks = build_flex_block_selected(
        context_tokens=context_tokens,
        suffix_tokens=int(k_suffix.shape[1]),
        flex_block_size=flex_block_size,
        ratio=ratio,
        device=q.device,
    )
    total_blocks = int(selected_blocks.numel())
    selected_idx = torch.nonzero(selected_blocks, as_tuple=False).flatten().to(torch.int32)

    def build_mask():
        # Keep kv_indices at a fixed max shape. Dynamic selected count is carried
        # by kv_num_blocks, which is closer to a decode-time routing update.
        kv_indices = torch.zeros((1, 1, 1, total_blocks), device=q.device, dtype=torch.int32)
        if int(selected_idx.numel()) > 0:
            kv_indices[0, 0, 0, : int(selected_idx.numel())] = selected_idx
        kv_num_blocks = torch.tensor(
            [[[int(selected_idx.numel())]]],
            device=q.device,
            dtype=torch.int32,
        )
        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=flex_block_size,
            seq_lengths=(1, int(k.shape[2])),
            compute_q_blocks=False,
        )

    block_mask, build_ms = wall_time_ms(build_mask)

    def call():
        return compiled_flex(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
        )

    return call, build_ms, int(selected_blocks[: math.ceil(context_tokens / flex_block_size)].sum().item())


def build_compact_hybrid_kv(prefix, selected, k_suffix, v_suffix, dtype):
    n_kv_heads = int(prefix["k_block_attn"].shape[0])
    n_blocks = int(prefix["block_valid_counts"].numel())
    k_parts = []
    v_parts = []
    bias_parts = []
    for block_idx in range(n_blocks):
        count = int(prefix["block_valid_counts"][block_idx].item())
        if count <= 0:
            continue
        if bool(selected[0, block_idx].item()):
            k_parts.append(prefix["k_block_attn"][:, block_idx, :count, :])
            v_parts.append(prefix["v_block_attn"][:, block_idx, :count, :])
            bias_parts.append(torch.zeros((count,), device=k_suffix.device, dtype=torch.float32))
        else:
            k_parts.append(prefix["k_bar"][:, block_idx : block_idx + 1, :].to(dtype))
            v_parts.append(prefix["v_bar"][:, block_idx : block_idx + 1, :].to(dtype))
            bias_parts.append(
                torch.full(
                    (1,),
                    math.log(max(count, 1)),
                    device=k_suffix.device,
                    dtype=torch.float32,
                )
            )
    if int(k_suffix.shape[1]) > 0:
        k_parts.append(k_suffix)
        v_parts.append(v_suffix)
        bias_parts.append(torch.zeros((int(k_suffix.shape[1]),), device=k_suffix.device, dtype=torch.float32))
    if not k_parts:
        head_dim = int(prefix["k_block_attn"].shape[-1])
        empty_k = torch.empty((n_kv_heads, 0, head_dim), device=k_suffix.device, dtype=dtype)
        return empty_k, empty_k, torch.empty((0,), device=k_suffix.device, dtype=torch.float32)
    return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1), torch.cat(bias_parts, dim=0)


def make_flex_compact_call(compiled_flex, q_grouped, k_compact, v_compact, score_bias):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    q = q_grouped.reshape(1, n_kv_heads * group_size, 1, head_dim)
    k = k_compact.unsqueeze(0)
    v = v_compact.unsqueeze(0)

    def score_mod(score, _b, _h, _q_idx, kv_idx):
        return score + score_bias[kv_idx]

    def call():
        return compiled_flex(
            q,
            k,
            v,
            score_mod=score_mod,
            scale=head_dim**-0.5,
            enable_gqa=True,
        )

    return call


def run_dense_compact_reference(q_grouped, k_compact, v_compact, score_bias):
    n_kv_heads, group_size, _, head_dim = q_grouped.shape
    q = q_grouped.reshape(n_kv_heads, group_size, head_dim).float()
    k = k_compact.float()
    v = v_compact.float()
    scores = torch.einsum("gsd,gtd->gst", q, k) * (head_dim**-0.5)
    scores = scores + score_bias.view(1, 1, -1)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("gst,gtd->gsd", weights, v)
    return out.reshape(1, n_kv_heads * group_size, 1, head_dim)


def run_sanity(args, compiled_flex, dtype, device):
    prefix = build_synthetic_prefix(
        n_kv_heads=args.kv_heads,
        context_tokens=args.sanity_context,
        block_size=args.condition_block_size,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    group_size = args.q_heads // args.kv_heads
    q_grouped = torch.randn((args.kv_heads, group_size, 1, args.head_dim), device=device, dtype=dtype)
    k_suffix = torch.randn((args.kv_heads, min(args.suffix_tokens, 16), args.head_dim), device=device, dtype=dtype)
    v_suffix = torch.randn_like(k_suffix)
    selected = build_dummy_selected(
        args.kv_heads,
        int(prefix["block_valid_counts"].numel()),
        0.25,
        device,
    )
    q = q_grouped.reshape(args.q_heads, args.head_dim).contiguous()
    triton_out = run_sparse_attention_dummy(
        q=q,
        prefix=prefix,
        selected=selected,
        k_suffix=k_suffix,
        v_suffix=v_suffix,
        group_size=group_size,
        head_dim=args.head_dim,
        block_size=args.condition_block_size,
        output_dtype=dtype,
    ).reshape(1, args.q_heads, 1, args.head_dim)
    k_compact, v_compact, bias = build_compact_hybrid_kv(prefix, selected, k_suffix, v_suffix, dtype)
    flex_out = make_flex_compact_call(compiled_flex, q_grouped, k_compact, v_compact, bias)()
    ref_out = run_dense_compact_reference(q_grouped, k_compact, v_compact, bias)
    torch.cuda.synchronize()
    flex_ref_err = float((flex_out.float() - ref_out.float()).abs().max().item())
    flex_triton_err = float((flex_out.float() - triton_out.float()).abs().max().item())

    try:
        block_call, _, _ = make_flex_block_call(
            compiled_flex,
            q_grouped,
            prefix,
            k_suffix,
            v_suffix,
            args.flex_block_size,
            0.5,
        )
        block_out = block_call()
        torch.cuda.synchronize()
        block_sparse_status = {
            "block_sparse_output_shape": list(block_out.shape),
        }
    except Exception as exc:
        block_sparse_status = {
            "block_sparse_error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }
    return {
        "compact_flex_vs_dense_max_error": flex_ref_err,
        "compact_flex_vs_triton_dummy_max_error": flex_triton_err,
        "compact_kv_tokens": int(k_compact.shape[1]),
        **block_sparse_status,
    }


def compile_flex(args):
    torch._dynamo.reset()
    return torch.compile(
        flex_attention,
        dynamic=not args.no_compile_dynamic,
        mode=args.compile_mode,
    )


def main():
    args = parse_args()
    set_seed(42)
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This benchmark requires a CUDA device.")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    compiled_flex = compile_flex(args)

    sanity = run_sanity(args, compiled_flex, dtype, device)
    print(json.dumps({"stage": "sanity", **sanity}), flush=True)
    if args.sanity_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    group_size = args.q_heads // args.kv_heads
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "sanity", **sanity}) + "\n")
        for context_tokens in args.contexts:
            # FlexDecoding currently specializes heavily on KV/block metadata.
            # Compile per context so one shape's generated graph does not leak
            # assumptions into the next shape in a sweep. Warmup absorbs compile
            # and autotune time; reported latency is steady-state CUDA event time.
            compiled_flex = compile_flex(args)
            prefix = build_synthetic_prefix(
                n_kv_heads=args.kv_heads,
                context_tokens=int(context_tokens),
                block_size=args.condition_block_size,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
            )
            q_grouped = torch.randn(
                (args.kv_heads, group_size, 1, args.head_dim),
                device=device,
                dtype=dtype,
            )
            q = q_grouped.reshape(args.q_heads, args.head_dim).contiguous()
            k_suffix = torch.randn(
                (args.kv_heads, int(args.suffix_tokens), args.head_dim),
                device=device,
                dtype=dtype,
            )
            v_suffix = torch.randn_like(k_suffix)

            full_row = {
                "stage": "full_sdpa_decode_attention",
                "context_tokens": int(context_tokens),
                "suffix_tokens": int(args.suffix_tokens),
                "latency_ms": cuda_time_ms(
                    lambda: run_full_sdpa_with_suffix(q_grouped, prefix, k_suffix, v_suffix),
                    args.warmup,
                    args.iters,
                ),
            }
            print(json.dumps(full_row), flush=True)
            handle.write(json.dumps(full_row) + "\n")

            try:
                flex_dense_call = make_flex_dense_call(
                    compiled_flex,
                    q_grouped,
                    prefix,
                    k_suffix,
                    v_suffix,
                )
                flex_dense_row = {
                    "stage": "flex_dense_decode_attention",
                    "context_tokens": int(context_tokens),
                    "suffix_tokens": int(args.suffix_tokens),
                    "compile_mode": args.compile_mode,
                    "latency_ms": cuda_time_ms(
                        flex_dense_call,
                        args.warmup,
                        args.iters,
                    ),
                }
            except Exception as exc:
                flex_dense_row = {
                    "stage": "flex_dense_decode_attention",
                    "context_tokens": int(context_tokens),
                    "suffix_tokens": int(args.suffix_tokens),
                    "compile_mode": args.compile_mode,
                    "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                }
            print(json.dumps(flex_dense_row), flush=True)
            handle.write(json.dumps(flex_dense_row) + "\n")

            for ratio in args.selected_ratios:
                selected = build_dummy_selected(
                    args.kv_heads,
                    int(prefix["block_valid_counts"].numel()),
                    ratio,
                    device,
                )
                triton_row = {
                    "stage": "triton_dummy_hybrid_attention",
                    "context_tokens": int(context_tokens),
                    "suffix_tokens": int(args.suffix_tokens),
                    "condition_block_size": int(args.condition_block_size),
                    "selected_ratio": float(ratio),
                    "compact_semantics": True,
                    "latency_ms": cuda_time_ms(
                        lambda selected=selected: run_sparse_attention_dummy(
                            q=q,
                            prefix=prefix,
                            selected=selected,
                            k_suffix=k_suffix,
                            v_suffix=v_suffix,
                            group_size=group_size,
                            head_dim=args.head_dim,
                            block_size=args.condition_block_size,
                            output_dtype=dtype,
                        ),
                        args.warmup,
                        args.iters,
                    ),
                }
                print(json.dumps(triton_row), flush=True)
                handle.write(json.dumps(triton_row) + "\n")

                (compact, build_ms) = wall_time_ms(
                    lambda selected=selected: build_compact_hybrid_kv(prefix, selected, k_suffix, v_suffix, dtype)
                )
                k_compact, v_compact, bias = compact
                flex_compact_call = make_flex_compact_call(
                    compiled_flex,
                    q_grouped,
                    k_compact,
                    v_compact,
                    bias,
                )
                flex_compact_row = {
                    "stage": "flex_compact_hybrid_attention",
                    "context_tokens": int(context_tokens),
                    "suffix_tokens": int(args.suffix_tokens),
                    "condition_block_size": int(args.condition_block_size),
                    "selected_ratio": float(ratio),
                    "compact_kv_tokens": int(k_compact.shape[1]),
                    "compact_build_ms": build_ms,
                    "compact_semantics": True,
                    "latency_ms": cuda_time_ms(
                        flex_compact_call,
                        args.warmup,
                        args.iters,
                    ),
                }
                print(json.dumps(flex_compact_row), flush=True)
                handle.write(json.dumps(flex_compact_row) + "\n")

                try:
                    block_call, mask_build_ms, selected_flex_blocks = make_flex_block_call(
                        compiled_flex,
                        q_grouped,
                        prefix,
                        k_suffix,
                        v_suffix,
                        args.flex_block_size,
                        ratio,
                    )
                    flex_block_row = {
                        "stage": "flex_block_sparse_token_attention",
                        "context_tokens": int(context_tokens),
                        "suffix_tokens": int(args.suffix_tokens),
                        "flex_block_size": int(args.flex_block_size),
                        "selected_ratio": float(ratio),
                        "selected_flex_blocks": selected_flex_blocks,
                        "block_mask_build_ms": mask_build_ms,
                        "compact_semantics": False,
                        "note": "Flex BlockMask skips token blocks; it does not add representatives for unselected prompt blocks.",
                        "latency_ms": cuda_time_ms(
                            block_call,
                            args.warmup,
                            args.iters,
                        ),
                    }
                except Exception as exc:
                    flex_block_row = {
                        "stage": "flex_block_sparse_token_attention",
                        "context_tokens": int(context_tokens),
                        "suffix_tokens": int(args.suffix_tokens),
                        "flex_block_size": int(args.flex_block_size),
                        "selected_ratio": float(ratio),
                        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                        "compact_semantics": False,
                    }
                print(json.dumps(flex_block_row), flush=True)
                handle.write(json.dumps(flex_block_row) + "\n")

                try:
                    block_call, mask_build_ms, selected_flex_blocks = make_flex_block_from_kv_blocks_call(
                        compiled_flex,
                        q_grouped,
                        prefix,
                        k_suffix,
                        v_suffix,
                        args.flex_block_size,
                        ratio,
                    )
                    flex_block_row = {
                        "stage": "flex_block_sparse_from_kv_blocks_attention",
                        "context_tokens": int(context_tokens),
                        "suffix_tokens": int(args.suffix_tokens),
                        "flex_block_size": int(args.flex_block_size),
                        "selected_ratio": float(ratio),
                        "selected_flex_blocks": selected_flex_blocks,
                        "block_mask_build_ms": mask_build_ms,
                        "compile_mode": args.compile_mode,
                        "compact_semantics": False,
                        "note": "Manually builds BlockMask.from_kv_blocks; skips token blocks but has no representative semantics.",
                        "latency_ms": cuda_time_ms(
                            block_call,
                            args.warmup,
                            args.iters,
                        ),
                    }
                except Exception as exc:
                    flex_block_row = {
                        "stage": "flex_block_sparse_from_kv_blocks_attention",
                        "context_tokens": int(context_tokens),
                        "suffix_tokens": int(args.suffix_tokens),
                        "flex_block_size": int(args.flex_block_size),
                        "selected_ratio": float(ratio),
                        "compile_mode": args.compile_mode,
                        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                        "compact_semantics": False,
                    }
                print(json.dumps(flex_block_row), flush=True)
                handle.write(json.dumps(flex_block_row) + "\n")


if __name__ == "__main__":
    main()
