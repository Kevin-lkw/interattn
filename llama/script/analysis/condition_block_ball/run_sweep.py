"""Full LongBench eps sweep with the diag_ell production (fused Triton) path.

Monkeypatches `core._run_condition_block_selection_stats` with the diag_ell
stats kernel and iterates `--eps-list` values, invoking `longbench.run_all`
once per eps (model reloads per eps; datasets iterate inside run_all with
fast-skip resume). Stats stay enabled so every record carries the realized
equivalent budget. Use a dedicated --output-root so box sweep results are not
mixed or fast-skipped against.

Example (one GPU's shard):

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=llama conda run -n nanogpt \
  python -m script.analysis.condition_block_ball.run_sweep \
  --datasets narrativeqa gov_report qmsum multi_news \
  --condition-block-size 32 --continue-on-error \
  --output-root llama/result/generate/diag_ell_sweep
"""

import sys

from ..condition_block_gen.longbench import run_all
from ..condition_block_gen.methods.condition_block_triton_impl import core
from .triton_selection import run_selection_stats_diag_ell

DEFAULT_EPS = ["0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5"]


def main():
    argv = sys.argv[1:]
    eps_values = DEFAULT_EPS
    if "--eps-list" in argv:
        idx = argv.index("--eps-list")
        end = idx + 1
        while end < len(argv) and not argv[end].startswith("--"):
            end += 1
        eps_values = argv[idx + 1 : end]
        argv = argv[:idx] + argv[end:]
    if not eps_values:
        raise ValueError("--eps-list needs at least one value")

    use_v3 = "--v3" in argv
    if use_v3:
        argv.remove("--v3")
        from .triton_selection_v3 import run_selection_stats_diag_ell_v3

        core._run_condition_block_selection_stats = run_selection_stats_diag_ell_v3
    else:
        core._run_condition_block_selection_stats = run_selection_stats_diag_ell
    print(f"[ball] diag_ell fused sweep (v3={use_v3}), eps={eps_values}", flush=True)
    for eps in eps_values:
        print(f"=== diag_ell sweep eps={eps} ===", flush=True)
        sys.argv = [
            sys.argv[0],
            *argv,
            "--method",
            "condition_block_triton",
            "--condition-eps",
            eps,
        ]
        run_all.main()


if __name__ == "__main__":
    main()
