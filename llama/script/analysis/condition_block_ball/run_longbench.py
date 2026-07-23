"""LongBench generation with ball-bound (or paired box) condition selection.

Wrapper around `condition_block_gen.longbench.run_all` that:

1. forces the legacy (non-fused) stage2 via CONDITION_BLOCK_LEGACY_STAGE2=1 so
   selection goes through the patchable `core._select_prompt_blocks`;
2. with --selection ball, monkeypatches that dispatcher with the ball version;
   with --selection box, leaves the original box selection so both runs share
   the identical attention pipeline;
3. forwards every other CLI argument to run_all unchanged (method fixed to
   condition_block_triton).

Stats stay enabled by default so the outputs record the realized budget.
"""

import os
import sys

from ..condition_block_gen.longbench import run_all
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .gen_selection import select_prompt_blocks_ball, select_prompt_blocks_diag_ell

SELECTIONS = {
    "ball": select_prompt_blocks_ball,
    "diag_ell": select_prompt_blocks_diag_ell,
    "box": None,
}


def main():
    argv = sys.argv[1:]
    selection = "ball"
    if "--selection" in argv:
        idx = argv.index("--selection")
        selection = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
    if selection not in SELECTIONS:
        raise ValueError(f"--selection must be one of {sorted(SELECTIONS)}, got {selection!r}")

    os.environ["CONDITION_BLOCK_LEGACY_STAGE2"] = "1"
    if os.environ.get("CONDITION_BLOCK_CUDA_GRAPH") == "1":
        raise ValueError("CUDA graph requires the fused path; unset CONDITION_BLOCK_CUDA_GRAPH.")
    if SELECTIONS[selection] is not None:
        core._select_prompt_blocks = SELECTIONS[selection]
    print(f"[ball] selection={selection}, legacy stage2 forced")

    sys.argv = [sys.argv[0], *argv, "--method", "condition_block_triton"]
    run_all.main()


if __name__ == "__main__":
    main()
