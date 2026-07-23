"""Decode-only latency for the diag_ell selection in the production fused path.

Wrapper around `condition_block_gen.longbench_v2_latency` that, with
`--selection diag_ell`, monkeypatches `core._run_condition_block_selection_stats`
with the diag_ell stats runner (same outputs; reads `k_bar`+`w`+`rho`). The
`w`/`rho` summaries are built lazily on the first (eager) decode step, so the
CUDA-graph capture replays the diag_ell kernel with fixed buffers. All other
CLI args are forwarded unchanged; the wrapper sets the recommended long-context
env (CUDA graph + post-prefill StaticCache; pass --skip-stats as CUDA graph
requires it).
"""

import os
import sys

from ..condition_block_gen import longbench_v2_latency
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .triton_selection import run_selection_stats_diag_ell


def main():
    argv = sys.argv[1:]
    selection = "box"
    if "--selection" in argv:
        idx = argv.index("--selection")
        selection = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
    if selection not in ("box", "diag_ell"):
        raise ValueError(f"--selection must be box or diag_ell, got {selection!r}")

    os.environ["CONDITION_BLOCK_CUDA_GRAPH"] = "1"
    os.environ["CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE"] = "1"
    if selection == "diag_ell":
        core._run_condition_block_selection_stats = run_selection_stats_diag_ell
    print(f"[ball] latency selection={selection}, CUDA graph + post-prefill StaticCache")

    sys.argv = [sys.argv[0], *argv]
    longbench_v2_latency.main()


if __name__ == "__main__":
    main()
