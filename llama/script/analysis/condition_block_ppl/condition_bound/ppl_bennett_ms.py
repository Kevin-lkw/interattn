"""Multisample PPL sweep with the Bennett condition.

Same monkeypatch as ppl_bennett.py, applied to the multisample suite
(analysis.condition_block_ppl.multisample.run_condition_block), so results are directly comparable
to the saved wikitext_n{1,20}/condition_block summaries. Accepts all
run_condition_block flags; pass a dedicated --output-root to avoid overwriting
the cosh baselines.
"""

from .ppl_bennett import _batched_hybrid_outputs_bennett

from analysis.condition_block_ppl import runner_cond_block as rcb
from analysis.condition_block_ppl.multisample import run_condition_block as ms

if __name__ == "__main__":
    rcb._batched_hybrid_outputs_for_queries = _batched_hybrid_outputs_bennett
    ms.main()
