"""Cold-L2 microbench: box vs diag_ell Triton selection-stats kernel.

Methodology mirrors `condition_block_stage_latency.py`: synthetic prompt
summaries, per-iteration L2 eviction with a 256 MiB buffer, CUDA-event timing,
two alternating A/B rounds. Before timing, the diag_ell kernel is
parity-checked: `s` must match the box kernel bitwise and `delta` must match
the eager diag_ell reference.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from ..condition_block_gen.condition_block_stage_latency import (
    build_synthetic_prefix,
    cuda_time_ms,
)
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import diag_ell_stats
from .triton_selection import run_selection_stats_diag_ell


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
    parser.add_argument("--warm-l2", action="store_true", help="Skip the per-iteration L2 flush.")
    parser.add_argument(
        "--w-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help=(
            "Storage dtype for the diag_ell w vector. bfloat16 rounds w UP by a "
            "2^-7 relative bump before casting, so the stored value is >= the "
            "exact w and the delta stays a strict upper bound."
        ),
    )
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ball_bench_selection.jsonl"))
    return parser.parse_args()


def eager_diag_ell_delta(q_grouped, prefix, scale):
    w, rho = diag_ell_stats(prefix)
    qf = q_grouped.float()
    qw = torch.sqrt(torch.einsum("gsqd,gbd->gsqb", qf.pow(2), w.float().pow(2)))
    return rho[:, None, None, :] * qw * scale


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    l2_flush = None
    if not args.warm_l2:
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
        q_grouped = torch.randn(
            (args.kv_heads, args.group_size, 1, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        if args.w_dtype == "bfloat16":
            os.environ["CONDITION_BLOCK_BALL_W_DTYPE"] = "bfloat16"
        diag_ell_stats(prefix)  # build w/rho outside the timed region

        ws_box, ws_diag = {}, {}
        _, s_box, delta_box, _, glob_box, _, _ = core._run_condition_block_selection_stats(
            q_grouped, prefix, workspace=ws_box
        )
        _, s_diag, delta_diag, _, glob_diag, _, _ = run_selection_stats_diag_ell(
            q_grouped, prefix, workspace=ws_diag
        )
        assert torch.equal(s_box, s_diag), "s must match the box kernel bitwise"
        ref_delta = eager_diag_ell_delta(q_grouped, prefix, args.head_dim**-0.5)
        ref_delta = ref_delta.reshape(args.kv_heads * args.group_size, n_blocks)
        delta_err = (delta_diag - ref_delta).abs().max().item()
        assert delta_err < 5e-3, f"delta mismatch vs eager reference: {delta_err}"

        def run_box():
            core._run_condition_block_selection_stats(q_grouped, prefix, workspace=ws_box)

        def run_diag():
            run_selection_stats_diag_ell(q_grouped, prefix, workspace=ws_diag)

        rounds = {"box": [], "diag_ell": []}
        for _ in range(2):  # alternate A/B to cancel drift
            rounds["box"].append(
                cuda_time_ms(run_box, args.warmup, args.iters, l2_flush=l2_flush) * 1e3
            )
            rounds["diag_ell"].append(
                cuda_time_ms(run_diag, args.warmup, args.iters, l2_flush=l2_flush) * 1e3
            )
        box_us = sum(rounds["box"]) / 2
        diag_us = sum(rounds["diag_ell"]) / 2
        record = {
            "context": context,
            "n_blocks": n_blocks,
            "cold_l2": not args.warm_l2,
            "box_us": box_us,
            "diag_ell_us": diag_us,
            "speedup": box_us / diag_us,
            "rounds": rounds,
            "delta_max_err_vs_eager": delta_err,
        }
        records.append(record)
        print(
            f"context={context} n_blocks={n_blocks} "
            f"box={box_us:.2f}us diag_ell={diag_us:.2f}us "
            f"speedup={record['speedup']:.3f}x (delta err {delta_err:.1e})"
        )
        del prefix, ws_box, ws_diag
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
