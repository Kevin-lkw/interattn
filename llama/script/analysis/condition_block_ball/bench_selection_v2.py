"""Parity + cold-L2 A/B: v1 (chunked) vs v2 (persistent) diag_ell stats kernel.

Parity gates before timing, per context:
- `s` and `delta` caches must match v1 bitwise (identical math on identical
  inputs; only the partial partitioning differs);
- the reduced global stats must match v1 within 1e-5 relative (the online
  in-program merge reorders the same exp/add operations).

Timing reports the stats kernel alone and stats+reduce (v2 shrinks the reduce
input from n_chunks to P).
"""

import argparse
import json
import os
from pathlib import Path

import torch

from ..condition_block_gen.condition_block_stage_latency import (
    build_synthetic_prefix,
    cuda_time_ms,
)
from .gen_selection import diag_ell_stats
from .triton_selection import run_selection_stats_diag_ell
from .triton_selection_v2 import run_selection_stats_diag_ell_v2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--with-reduce", action="store_true", help="Time stats+reduce instead of stats alone.")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ball_bench_selection_v2.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    os.environ["CONDITION_BLOCK_BALL_W_DTYPE"] = "bfloat16"
    l2_flush = torch.empty(
        (args.l2_flush_mib * 1024 * 1024 // 4,), device=device, dtype=torch.float32
    )
    records = []
    for context in args.contexts:
        prefix = build_synthetic_prefix(
            n_kv_heads=args.kv_heads,
            context_tokens=context,
            block_size=args.block_size,
            head_dim=args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        n_blocks = int(prefix["block_valid_counts"].numel())
        prefix["valid_token"] = (
            torch.arange(args.block_size, device=device)[None, :]
            < prefix["block_valid_counts"][:, None]
        )
        prefix["k_bar"] = prefix["k_bar"].to(torch.bfloat16)
        q_grouped = torch.randn(
            (args.kv_heads, args.group_size, 1, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        diag_ell_stats(prefix)

        ws_v1, ws_v2 = {}, {}
        _, s1, d1, _, glob1, _, nc1 = run_selection_stats_diag_ell(
            q_grouped, prefix, workspace=ws_v1
        )
        _, s2, d2, _, glob2, _, nc2 = run_selection_stats_diag_ell_v2(
            q_grouped, prefix, workspace=ws_v2
        )
        # Different tile/warp shapes reorder the FP32 d-reduction, so parity is
        # last-ulp, not bitwise (v1 itself differs across its warp autotune).
        s_err = (s1 - s2).abs().max().item()
        d_err = (d1 - d2).abs().max().item()
        assert torch.allclose(s1, s2, rtol=1e-5, atol=1e-4), f"v2 s mismatch: max abs {s_err}"
        assert torch.allclose(d1, d2, rtol=1e-5, atol=1e-4), f"v2 delta mismatch: max abs {d_err}"
        glob_err = (
            ((glob1 - glob2).abs() / glob1.abs().clamp_min(1e-30)).max().item()
        )
        assert glob_err < 1e-5, f"v2 global stats mismatch: rel err {glob_err}"

        reduce_globals = bool(args.with_reduce)

        def run_v1():
            run_selection_stats_diag_ell(
                q_grouped, prefix, workspace=ws_v1, reduce_globals=reduce_globals
            )

        def run_v2():
            run_selection_stats_diag_ell_v2(
                q_grouped, prefix, workspace=ws_v2, reduce_globals=reduce_globals
            )

        rounds = {"v1": [], "v2": []}
        for _ in range(2):
            rounds["v1"].append(cuda_time_ms(run_v1, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
            rounds["v2"].append(cuda_time_ms(run_v2, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
        v1_us = sum(rounds["v1"]) / 2
        v2_us = sum(rounds["v2"]) / 2
        record = {
            "context": context,
            "n_blocks": n_blocks,
            "with_reduce": reduce_globals,
            "v2_p": nc2,
            "v2_block": int(os.environ.get("CONDITION_BLOCK_SELECT_V2_BLOCK", "32")),
            "v2_stages": int(os.environ.get("CONDITION_BLOCK_SELECT_V2_STAGES", "3")),
            "v2_warps": int(os.environ.get("CONDITION_BLOCK_SELECT_V2_WARPS", "4")),
            "v1_us": v1_us,
            "v2_us": v2_us,
            "speedup": v1_us / v2_us,
            "glob_rel_err": glob_err,
            "rounds": rounds,
        }
        records.append(record)
        print(
            f"context={context} n_blocks={n_blocks} P={nc2} "
            f"v1={v1_us:.2f}us v2={v2_us:.2f}us speedup={record['speedup']:.3f}x "
            f"(glob err {glob_err:.1e})"
        )
        del prefix, ws_v1, ws_v2
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
