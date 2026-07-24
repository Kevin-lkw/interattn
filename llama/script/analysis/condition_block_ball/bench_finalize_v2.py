"""Parity + cold-L2 A/B: production fused stage2 vs the persistent finalize v2.

Both sides run the v3 selection-stats kernel, so the A/B isolates
finalize+reduce. Gates before timing, per context and per eps:

- `selected` block sets bitwise identical (same s/delta/globals, same
  per-chunk decision math);
- attention output allclose (rtol 1e-4 / atol 1e-5 in FP32; the persistent
  span regroups the online-softmax merge order);
- run twice with eps 0.1 (sparse pages) and a stress eps that selects many
  pages, plus a 128-token suffix, so the page loop and the distributed
  suffix path are both exercised.

Timing is the full fused call (stats v3 + finalize + reduce), cold L2.
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
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import diag_ell_stats
from .triton_selection_v3 import run_selection_stats_diag_ell_v3
from .triton_finalize_v2 import decode_output_fused_v2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--suffix-len", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--stress-eps", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument(
        "--impl",
        default="v2",
        choices=["v2", "split"],
        help="Candidate implementation to compare against the production kernel.",
    )
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ball_bench_finalize_v2.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    os.environ["CONDITION_BLOCK_BALL_W_DTYPE"] = "bfloat16"
    core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v3
    global decode_output_fused_v2
    if args.impl == "split":
        from .triton_finalize_v3 import decode_output_fused_split

        decode_output_fused_v2 = decode_output_fused_split
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
        prefix["v_bar"] = prefix["v_bar"].to(torch.bfloat16)
        diag_ell_stats(prefix)
        q_grouped = torch.randn(
            (args.kv_heads, args.group_size, 1, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        k_suffix = torch.randn(
            (args.kv_heads, args.suffix_len, args.head_dim), device=device, dtype=torch.bfloat16
        )
        v_suffix = torch.randn_like(k_suffix)
        suffix_len_dev = torch.tensor([args.suffix_len], device=device, dtype=torch.int32)

        for eps in (args.eps, args.stress_eps):
            ws_ref, ws_v2 = {}, {}
            call_kwargs = dict(
                q_grouped=q_grouped, prompt_prefix=prefix, k_suffix=k_suffix,
                v_suffix=v_suffix, suffix_len_dev=suffix_len_dev, eps=eps,
                page_size=args.block_size, store_selected=True,
                output_dtype=torch.float32,
            )
            out_ref, sel_ref = core._condition_block_decode_output_fused_triton(
                workspace=ws_ref, **call_kwargs
            )
            out_v2, sel_v2 = decode_output_fused_v2(workspace=ws_v2, **call_kwargs)
            assert torch.equal(sel_ref, sel_v2), f"selected sets differ at eps={eps}"
            err = (out_ref - out_v2).abs().max().item()
            # Regrouping the online-softmax merges (span-sequential vs per-chunk
            # + reduce) reorders thousands of FP32 exp-weighted adds; observed
            # noise is ~1e-5, two orders below the BF16 output granularity.
            assert torch.allclose(out_ref, out_v2, rtol=1e-4, atol=1e-4), (
                f"output mismatch at eps={eps}: max abs {err}"
            )
            n_sel = int(sel_ref[:, 0].sum().item())
            print(
                f"[parity {context} eps={eps}] selected={n_sel} blocks/head-row, "
                f"out max err={err:.2e} OK"
            )

        record = {
            "context": context,
            "n_blocks": n_blocks,
            "fin_p": int(os.environ.get("CONDITION_BLOCK_FIN_V2_P", "48")),
            "fin_block": int(os.environ.get("CONDITION_BLOCK_FIN_V2_BLOCK", "32")),
            "fin_stages": int(os.environ.get("CONDITION_BLOCK_FIN_V2_STAGES", "2")),
            "fin_warps": int(os.environ.get("CONDITION_BLOCK_FIN_V2_WARPS", "4")),
        }
        if not args.parity_only:
            ws_ref, ws_v2 = {}, {}

            time_kwargs = dict(
                q_grouped=q_grouped, prompt_prefix=prefix, k_suffix=k_suffix,
                v_suffix=v_suffix, suffix_len_dev=suffix_len_dev, eps=args.eps,
                page_size=args.block_size, store_selected=False,
                output_dtype=torch.float32,
            )

            def run_ref():
                core._condition_block_decode_output_fused_triton(workspace=ws_ref, **time_kwargs)

            def run_v2():
                decode_output_fused_v2(workspace=ws_v2, **time_kwargs)

            rounds = {"ref": [], "v2": []}
            for _ in range(2):
                rounds["ref"].append(cuda_time_ms(run_ref, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
                rounds["v2"].append(cuda_time_ms(run_v2, args.warmup, args.iters, l2_flush=l2_flush) * 1e3)
            record["ref_us"] = sum(rounds["ref"]) / 2
            record["v2_us"] = sum(rounds["v2"]) / 2
            record["speedup"] = record["ref_us"] / record["v2_us"]
            record["rounds"] = rounds
            print(
                f"context={context} fused-call ref={record['ref_us']:.2f}us "
                f"v2={record['v2_us']:.2f}us speedup={record['speedup']:.3f}x"
            )
        records.append(record)
        del prefix, ws_ref, ws_v2
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
