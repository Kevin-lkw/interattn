"""Multisample PPL sweep with the ball-bound condition.

Same monkeypatch pattern as `condition_bound/ppl_bennett_ms.py`: replace only
`runner_cond_block._batched_hybrid_outputs_for_queries` with the ball-delta
version, then run the multisample suite unchanged. Results are directly
comparable to the saved box-condition summaries (same protocol, budgets are
measured, not nominal). Pass a dedicated --output-root so the box baselines are
not overwritten.
"""

from ..condition_block_ppl import runner_cond_block as rcb
from ..condition_block_ppl.multisample import run_condition_block as ms
from .ppl_condition import _batched_hybrid_outputs_ball

if __name__ == "__main__":
    print("[ball] using ball-bound delta; --delta-mode is ignored")
    rcb._batched_hybrid_outputs_for_queries = _batched_hybrid_outputs_ball
    ms.main()
