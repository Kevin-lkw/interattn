"""Kernel-level decode attribution with box or diag_ell selection.

Wrapper around `condition_block_gen.condition_block_decode_attribution` (the
methodology behind the impl README's "vs full attention" table: clean decode
wall from an unprofiled run, per-kernel device-side times from a separate
profiled run, CUDA-graph decode). `--selection diag_ell` monkeypatches the
production stats kernel; everything else is forwarded unchanged.
"""

import os
import sys

from ..condition_block_gen import condition_block_decode_attribution
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .triton_selection import run_selection_stats_diag_ell


def main():
    argv = sys.argv[1:]
    selection = "box"
    if "--selection" in argv:
        idx = argv.index("--selection")
        selection = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
    if selection not in ("box", "diag_ell", "diag_ell_v2", "diag_ell_v3"):
        raise ValueError(
            f"--selection must be box, diag_ell, diag_ell_v2 or diag_ell_v3, got {selection!r}"
        )

    os.environ["CONDITION_BLOCK_SKIP_STATS"] = "1"
    os.environ["CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE"] = "1"
    os.environ["CONDITION_BLOCK_CUDA_GRAPH"] = "1"
    if selection == "diag_ell":
        core._run_condition_block_selection_stats = run_selection_stats_diag_ell
    elif selection == "diag_ell_v2":
        from .triton_selection_v2 import run_selection_stats_diag_ell_v2

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v2
    elif selection == "diag_ell_v3":
        from .triton_selection_v3 import run_selection_stats_diag_ell_v3

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v3
    print(f"[ball] attribution selection={selection}, CUDA graph + StaticCache + stats off")

    sys.argv = [sys.argv[0], *argv]
    condition_block_decode_attribution.main()


if __name__ == "__main__":
    main()
