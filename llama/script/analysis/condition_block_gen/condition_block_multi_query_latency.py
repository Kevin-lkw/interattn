"""Measure the selection-IO upper bound from processing multiple queries at once.

This is a synthetic stage benchmark, not a generation implementation.  It
answers whether speculative/multi-token scheduling is worth integrating by
loading each prompt-summary tile once for K query positions.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from .condition_block_stage_latency import build_synthetic_prefix, cuda_time_ms
from .methods.condition_block_triton_impl.core import (
    _SELECT_CHUNK,
    _condition_block_selection_reduce_kernel,
    _run_condition_block_selection_stats,
)
from .methods.condition_block_triton_impl.selection_multi_query import (
    condition_block_selection_stats_multi_query_kernel,
)
from ..runner_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic multi-query condition-selection IO upper bound."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", nargs="+", type=int, default=[65536, 131072])
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--multi-warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/condition_block_multi_query_latency.jsonl"),
    )
    return parser.parse_args()


def _empty(workspace, key, shape, device):
    value = workspace.get(key)
    if value is None or tuple(value.shape) != tuple(shape):
        value = torch.empty(shape, device=device, dtype=torch.float32)
        workspace[key] = value
    return value


def run_multi_query_selection(q, prefix, workspace, num_warps):
    """Run original condition statistics for q shaped [H, K, G, D]."""
    n_kv_heads, queries, group_size, head_dim = q.shape
    query_rows = queries * group_size
    n_blocks = int(prefix["block_valid_counts"].numel())
    n_chunks = triton.cdiv(n_blocks, _SELECT_CHUNK)
    rows = n_kv_heads * query_rows
    q_flat = q.reshape(rows, head_dim).contiguous()
    s_cache = _empty(workspace, "s", (rows, n_blocks), q.device)
    delta_cache = _empty(workspace, "delta", (rows, n_blocks), q.device)
    partial = _empty(workspace, "partial", (4, rows, n_chunks), q.device)
    global_stats = _empty(workspace, "global", (4, rows), q.device)
    block_d = triton.next_power_of_2(head_dim)
    if block_d != head_dim:
        raise ValueError("multi-query TMA prototype requires power-of-two head_dim")
    bounds_desc = TensorDescriptor.from_tensor(
        prefix["k_bounds"],
        block_shape=[1, _SELECT_CHUNK, 2 * block_d],
    )
    condition_block_selection_stats_multi_query_kernel[(n_kv_heads, n_chunks)](
        q_flat,
        prefix["k_bar"].contiguous(),
        bounds_desc,
        prefix["block_valid_counts"].contiguous(),
        s_cache,
        delta_cache,
        partial[0],
        partial[1],
        partial[2],
        partial[3],
        n_blocks,
        n_chunks,
        head_dim,
        head_dim**-0.5,
        QUERY_ROWS=query_rows,
        BLOCK_R=triton.next_power_of_2(query_rows),
        BLOCK_B=_SELECT_CHUNK,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    _condition_block_selection_reduce_kernel[(rows,)](
        partial[0],
        partial[1],
        partial[2],
        partial[3],
        global_stats[0],
        global_stats[1],
        global_stats[2],
        global_stats[3],
        n_chunks,
        BLOCK_C=triton.next_power_of_2(n_chunks),
        num_warps=4,
    )
    return s_cache, delta_cache, partial, global_stats


def sanity_against_single_query(q, prefix, multi_result, queries):
    n_kv_heads, _, group_size, _ = q.shape
    n_blocks = int(prefix["block_valid_counts"].numel())
    multi_s, multi_delta, _multi_partial, multi_global = multi_result
    multi_s = multi_s.view(n_kv_heads, queries, group_size, n_blocks)
    multi_delta = multi_delta.view(n_kv_heads, queries, group_size, n_blocks)
    multi_global = multi_global.view(4, n_kv_heads, queries, group_size)
    checks = []
    max_s_abs = 0.0
    max_delta_abs = 0.0
    max_global_abs = 0.0
    for query_idx in range(queries):
        single = _run_condition_block_selection_stats(
            q[:, query_idx, :, None, :],
            prefix,
            workspace={},
            reduce_globals=True,
        )
        single_s = single[1].view(n_kv_heads, group_size, n_blocks)
        single_delta = single[2].view(n_kv_heads, group_size, n_blocks)
        single_global = single[4].view(4, n_kv_heads, group_size)
        query_s = multi_s[:, query_idx]
        query_delta = multi_delta[:, query_idx]
        query_global = multi_global[:, :, query_idx]
        checks.append(
            torch.equal(query_s, single_s)
            and torch.equal(query_delta, single_delta)
            and torch.equal(query_global, single_global)
        )
        max_s_abs = max(max_s_abs, float((query_s - single_s).abs().max().item()))
        max_delta_abs = max(
            max_delta_abs,
            float((query_delta - single_delta).abs().max().item()),
        )
        max_global_abs = max(
            max_global_abs,
            float((query_global - single_global).abs().max().item()),
        )
    return {
        "bitwise_exact": all(checks),
        "max_s_abs": max_s_abs,
        "max_delta_abs": max_delta_abs,
        "max_global_abs": max_global_abs,
    }


def main():
    args = parse_args()
    if args.q_heads % args.kv_heads:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if min(args.queries) < 1:
        raise ValueError("--queries values must be >= 1")
    os.environ["CONDITION_BLOCK_MIXED_SUMMARIES"] = "1"
    os.environ["CONDITION_BLOCK_TMA_BOUNDS"] = "1"
    os.environ["CONDITION_BLOCK_TMA_PERSIST_CHUNKS"] = "1"
    set_seed(42)
    device = torch.device(args.device)
    group_size = args.q_heads // args.kv_heads
    l2_flush = torch.empty(
        args.l2_flush_mib * 1024 * 1024,
        device=device,
        dtype=torch.uint8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for context_tokens in args.contexts:
            prefix = build_synthetic_prefix(
                n_kv_heads=args.kv_heads,
                context_tokens=context_tokens,
                block_size=args.block_size,
                head_dim=args.head_dim,
                dtype=torch.bfloat16,
                device=device,
                mixed_summaries=True,
            )
            max_queries = max(args.queries)
            q_all = torch.randn(
                (args.kv_heads, max_queries, group_size, args.head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            single_workspace = {}
            single_cold_ms = cuda_time_ms(
                lambda: _run_condition_block_selection_stats(
                    q_all[:, 0, :, None, :],
                    prefix,
                    workspace=single_workspace,
                    reduce_globals=True,
                ),
                args.warmup,
                args.iters,
                l2_flush,
            )
            for queries in args.queries:
                q = q_all[:, :queries].contiguous()
                sequential_workspaces = [{} for _ in range(queries)]
                sequential_group_ms = cuda_time_ms(
                    lambda: [
                        _run_condition_block_selection_stats(
                            q[:, idx, :, None, :],
                            prefix,
                            workspace=sequential_workspaces[idx],
                            reduce_globals=True,
                        )
                        for idx in range(queries)
                    ],
                    args.warmup,
                    args.iters,
                    l2_flush,
                )
                for warps in args.multi_warps:
                    workspace = {}
                    result = run_multi_query_selection(q, prefix, workspace, warps)
                    torch.cuda.synchronize()
                    sanity = sanity_against_single_query(q, prefix, result, queries)
                    multi_ms = cuda_time_ms(
                        lambda: run_multi_query_selection(q, prefix, workspace, warps),
                        args.warmup,
                        args.iters,
                        l2_flush,
                    )
                    cold_k_estimate = queries * single_cold_ms
                    row = {
                        "context_tokens": context_tokens,
                        "block_size": args.block_size,
                        "queries": queries,
                        "multi_warps": warps,
                        "single_query_cold_ms": single_cold_ms,
                        "k_single_cold_estimate_ms": cold_k_estimate,
                        "sequential_k_after_one_flush_ms": sequential_group_ms,
                        "multi_query_group_ms": multi_ms,
                        "multi_query_ms_per_query": multi_ms / queries,
                        "speedup_vs_k_cold": cold_k_estimate / multi_ms,
                        "speedup_vs_sequential_warm_group": sequential_group_ms / multi_ms,
                        **sanity,
                    }
                    print(json.dumps(row), flush=True)
                    handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
