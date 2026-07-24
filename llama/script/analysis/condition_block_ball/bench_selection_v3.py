"""Soundness + parity + cold-L2 A/B for the tensor-core v3 stats kernel.

Gates (per context, before any timing):
1. Soundness: kernel `delta` >= exact dense per-token deviation from the
   kernel's own `s` center (the strict-bound property, checked directly).
2. `s` matches v1 within FP32 reduction-order tolerance.
3. `delta` matches a torch reference of the v3 formula (BF16 q^2 x stored w2,
   FP32 accumulate, (1+2^-8) inflation).
4. Reduced global z-stats match v1; selection-decision probe reports how many
   (row, block) decisions flip at eps=0.1 (v1's round-up w and v3's
   round-to-nearest w2 are different valid weights, so borderline decisions
   can flip in either direction by ~2^-8 relative delta).

`--parity-only` skips timing (usable on a shared GPU; timing needs exclusive).
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
from .gen_selection import _select_prompt_blocks_with_delta, diag_ell_stats
from .triton_selection import run_selection_stats_diag_ell
from .triton_selection_v3 import (
    _DELTA_INFLATION,
    diag_ell_v3_stats,
    run_selection_stats_diag_ell_v3,
)


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
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ball_bench_selection_v3.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    os.environ["CONDITION_BLOCK_BALL_W_DTYPE"] = "bfloat16"
    scale = args.head_dim**-0.5
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
        w2, rho3 = diag_ell_v3_stats(prefix)

        ws_v1, ws_v3 = {}, {}
        _, s1, d1, _, glob1, _, _ = run_selection_stats_diag_ell(
            q_grouped, prefix, workspace=ws_v1
        )
        _, s3, d3, _, glob3, _, num_p = run_selection_stats_diag_ell_v3(
            q_grouped, prefix, workspace=ws_v3
        )

        # 1. Soundness: dense exact deviation from the kernel's own center.
        qf = q_grouped.reshape(-1, args.head_dim).float()  # (rows, d)
        k_pages = prefix["k_block_attn"].float()  # (kv, nb, bs, d)
        rows = qf.shape[0]
        qh = qf.view(args.kv_heads, args.group_size, args.head_dim)
        scores = torch.einsum("kgd,kbtd->kgbt", qh, k_pages) * scale
        scores = scores.reshape(rows, n_blocks, args.block_size)
        dev = (scores - s3.unsqueeze(-1)).abs()
        dev = dev.masked_fill(~prefix["valid_token"].unsqueeze(0), 0.0)
        dev = dev.amax(dim=-1)
        violations = int((d3 < dev - 1e-4).sum().item())
        assert violations == 0, f"v3 soundness: {violations} delta < exact deviation"

        # 2/3. Parity vs v1 s and vs the torch v3-delta reference.
        s_err = (s1 - s3).abs().max().item()
        assert torch.allclose(s1, s3, rtol=1e-5, atol=1e-4), f"v3 s mismatch: {s_err}"
        q2_bf = (qf * qf).to(torch.bfloat16).float().view(
            args.kv_heads, args.group_size, args.head_dim
        )
        qw2_ref = torch.einsum("kgd,kbd->kgb", q2_bf, w2.float())
        ref_d3 = (
            rho3[:, None, :] * qw2_ref.clamp_min(0.0).sqrt() * scale * _DELTA_INFLATION
        ).reshape(rows, n_blocks)
        d_err = (d3 - ref_d3).abs().max().item()
        assert torch.allclose(d3, ref_d3, rtol=1e-3, atol=1e-4), f"v3 delta vs ref: {d_err}"

        # 4. Globals (z half only: same s) + selection-decision probe.
        zg_err = max(
            (glob1[0] - glob3[0]).abs().max().item(),
            ((glob1[1] - glob3[1]).abs() / glob1[1].abs().clamp_min(1e-30)).max().item(),
        )
        assert zg_err < 1e-4, f"v3 z-globals mismatch: {zg_err}"

        def v1_delta_fn(qg, pfx, sc):
            w, rho = diag_ell_stats(pfx)
            qw = torch.sqrt(
                torch.einsum("gsqd,gbd->gsqb", qg.float().pow(2), w.float().pow(2)).clamp_min(0.0)
            )
            return (rho[:, None, None, :] * qw / sc).to(qg.dtype)

        def v3_delta_fn(qg, pfx, sc):
            q2 = (qg.float().pow(2)).to(torch.bfloat16).float()
            qw2b = torch.einsum("gsqd,gbd->gsqb", q2, w2.float()).clamp_min(0.0)
            return (rho3[:, None, None, :] * qw2b.sqrt() * _DELTA_INFLATION / sc).to(qg.dtype)

        sel1 = _select_prompt_blocks_with_delta(q_grouped, prefix, 0.1, v1_delta_fn)[0]
        sel3 = _select_prompt_blocks_with_delta(q_grouped, prefix, 0.1, v3_delta_fn)[0]
        gained = int((sel3 & ~sel1).sum().item())
        lost = int((sel1 & ~sel3).sum().item())
        n1 = int(sel1.sum().item())
        print(
            f"[parity {context}] sound (0 violations), s_err={s_err:.1e}, "
            f"d_ref_err={d_err:.1e}, selection: {n1} blocks, +{gained}/-{lost} flips"
        )

        record = {
            "context": context,
            "n_blocks": n_blocks,
            "v3_p": num_p,
            "v3_block": int(os.environ.get("CONDITION_BLOCK_SELECT_V3_BLOCK", "64")),
            "v3_stages": int(os.environ.get("CONDITION_BLOCK_SELECT_V3_STAGES", "3")),
            "v3_warps": int(os.environ.get("CONDITION_BLOCK_SELECT_V3_WARPS", "4")),
            "selection_flips": {"gained": gained, "lost": lost, "v1_selected": n1},
        }
        if not args.parity_only:
            def run_v1():
                run_selection_stats_diag_ell(q_grouped, prefix, workspace=ws_v1)

            def run_v3():
                run_selection_stats_diag_ell_v3(q_grouped, prefix, workspace=ws_v3)

            rounds = {"v1": [], "v3": []}
            for _ in range(2):
                rounds["v1"].append(cuda_time_ms(run_v1, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
                rounds["v3"].append(cuda_time_ms(run_v3, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
            record["v1_us"] = sum(rounds["v1"]) / 2
            record["v3_us"] = sum(rounds["v3"]) / 2
            record["speedup"] = record["v1_us"] / record["v3_us"]
            record["rounds"] = rounds
            print(
                f"context={context} n_blocks={n_blocks} P={num_p} "
                f"v1={record['v1_us']:.2f}us v3={record['v3_us']:.2f}us "
                f"speedup={record['speedup']:.3f}x"
            )
        records.append(record)
        del prefix, ws_v1, ws_v3, scores, dev
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
