"""Synthetic correctness and cold-L2 latency benchmark for Tensor Core selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ..core import _run_condition_block_selection_stats
from .kernel import run_tensorcore_selection_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["ieee", "tf32x3", "tf32"],
        choices=["ieee", "tf32x3", "tf32"],
    )
    parser.add_argument("--warps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/condition_block_tensorcore_selection.jsonl"),
    )
    return parser.parse_args()


def build_prefix(*, context_tokens, block_size, kv_heads, head_dim, device):
    """Build range summaries directly without allocating full prompt KV."""

    n_blocks = math.ceil(context_tokens / block_size)
    k_bar = torch.randn(
        (kv_heads, n_blocks, head_dim), device=device, dtype=torch.float32
    ) * 0.25
    radius_hi = torch.rand_like(k_bar) * 0.75
    radius_lo = torch.rand_like(k_bar) * 0.75
    k_max = k_bar + radius_hi
    k_min = k_bar - radius_lo
    counts = torch.full((n_blocks,), block_size, device=device, dtype=torch.long)
    if context_tokens % block_size:
        counts[-1] = context_tokens % block_size
    v_norm_max = torch.rand((kv_heads, n_blocks), device=device) + 0.5
    return {
        "k_bar": k_bar,
        "k_max": k_max,
        "k_min": k_min,
        "block_valid_counts": counts,
        "v_norm_max": v_norm_max,
        "v_norm_all": v_norm_max.amax(dim=-1),
    }


def condition_from_stats(s, delta, prefix, *, group_size):
    kv_heads = int(prefix["k_bar"].shape[0])
    n_blocks = int(prefix["block_valid_counts"].numel())
    s = s.reshape(kv_heads, group_size, n_blocks)
    delta = delta.reshape(kv_heads, group_size, n_blocks)
    counts = prefix["block_valid_counts"].clamp_min(1).float()
    z = s + counts.log()[None, None, :]
    p = torch.softmax(z, dim=-1)
    cosh_delta = torch.cosh(delta)
    denom = (p * cosh_delta).sum(dim=-1).clamp_min(1e-30)
    b_c = prefix["v_norm_max"][:, None, :]
    b_all = prefix["v_norm_all"][:, None]
    condition = p * (
        2.0 * b_all[:, :, None] * (cosh_delta - 1.0) / denom[:, :, None]
        + 2.0 * b_c * torch.tanh(delta / 2.0)
    )
    return condition.mean(dim=1)


def threshold_mismatch_sweep(condition, condition_ref):
    """Compare routing at fixed reference quantiles, including boundary cases."""

    rows = []
    flat_ref = condition_ref.flatten()
    for selected_ratio in (0.001, 0.01, 0.05, 0.1, 0.5):
        threshold = torch.quantile(flat_ref, 1.0 - selected_ratio)
        selected_ref = condition_ref > threshold
        selected = condition > threshold
        rows.append(
            {
                "target_ratio": selected_ratio,
                "threshold": float(threshold.item()),
                "selected_ref": int(selected_ref.sum().item()),
                "mismatch": int((selected != selected_ref).sum().item()),
            }
        )
    return rows


def error_metrics(actual, expected):
    diff = (actual - expected).abs()
    denom = expected.abs().clamp_min(1e-7)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def cuda_time_ms(fn, *, warmup, iters, l2_flush):
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush.add_(1)
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        if l2_flush is not None:
            l2_flush.add_(1)
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / iters


def main():
    args = parse_args()
    if args.block_size != 32:
        raise ValueError("This experiment currently targets block_size=32.")
    if args.group_size != 4 or args.head_dim != 128:
        raise ValueError("This experiment targets group_size=4 and head_dim=128.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    l2_flush = None
    if args.cold_l2:
        l2_flush = torch.empty(
            args.l2_flush_mib * 1024 * 1024,
            device=device,
            dtype=torch.uint8,
        )

    with args.output.open("w", encoding="utf-8") as handle:
        for context_tokens in args.contexts:
            prefix = build_prefix(
                context_tokens=context_tokens,
                block_size=args.block_size,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                device=device,
            )
            q_grouped = torch.randn(
                (args.kv_heads, args.group_size, 1, args.head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            baseline_workspace = {}
            baseline = _run_condition_block_selection_stats(
                q_grouped,
                prefix,
                baseline_workspace,
                reduce_globals=True,
            )
            _, s_ref, delta_ref, _, global_ref, n_blocks, _ = baseline
            condition_ref = condition_from_stats(
                s_ref,
                delta_ref,
                prefix,
                group_size=args.group_size,
            )
            selected_ref = condition_ref > args.eps
            baseline_ms = cuda_time_ms(
                lambda: _run_condition_block_selection_stats(
                    q_grouped,
                    prefix,
                    baseline_workspace,
                    reduce_globals=True,
                ),
                warmup=args.warmup,
                iters=args.iters,
                l2_flush=l2_flush,
            )
            baseline_row = {
                "context_tokens": context_tokens,
                "n_blocks": n_blocks,
                "kind": "baseline",
                "cold_l2": args.cold_l2,
                "latency_ms": baseline_ms,
            }
            print(json.dumps(baseline_row), flush=True)
            handle.write(json.dumps(baseline_row) + "\n")

            for precision in args.precisions:
                for num_warps in args.warps:
                    workspace = {}
                    result = run_tensorcore_selection_stats(
                        q_grouped,
                        prefix,
                        workspace,
                        precision=precision,
                        reduce_globals=True,
                        num_warps=num_warps,
                    )
                    _, s, delta, _, global_stats, _, _ = result
                    torch.cuda.synchronize()
                    condition = condition_from_stats(
                        s,
                        delta,
                        prefix,
                        group_size=args.group_size,
                    )
                    selected = condition > args.eps
                    mismatch = int((selected != selected_ref).sum().item())
                    latency_ms = cuda_time_ms(
                        lambda precision=precision, num_warps=num_warps, workspace=workspace: run_tensorcore_selection_stats(
                            q_grouped,
                            prefix,
                            workspace,
                            precision=precision,
                            reduce_globals=True,
                            num_warps=num_warps,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        l2_flush=l2_flush,
                    )
                    row = {
                        "context_tokens": context_tokens,
                        "n_blocks": n_blocks,
                        "kind": "tensorcore",
                        "precision": precision,
                        "num_warps": num_warps,
                        "cold_l2": args.cold_l2,
                        "latency_ms": latency_ms,
                        "speedup": baseline_ms / latency_ms,
                        "s_error": error_metrics(s, s_ref),
                        "delta_error": error_metrics(delta, delta_ref),
                        "global_error": error_metrics(global_stats, global_ref),
                        "selected_mismatch": mismatch,
                        "selected_ref": int(selected_ref.sum().item()),
                        "selected_total": int(selected.numel()),
                        "threshold_sweep": threshold_mismatch_sweep(
                            condition,
                            condition_ref,
                        ),
                    }
                    print(json.dumps(row), flush=True)
                    handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
