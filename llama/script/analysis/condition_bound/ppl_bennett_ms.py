"""Multisample PPL sweep with the Bennett condition.

Same monkeypatch as ppl_bennett.py, applied to the multisample suite
(analysis.multisample.run_condition_block), so results are directly comparable
to the saved wikitext_n{1,20}/condition_block summaries. Accepts all
run_condition_block flags; pass a dedicated --output-root to avoid overwriting
the cosh baselines.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppl_bennett import _batched_hybrid_outputs_bennett  # noqa: E402

from analysis import runner_cond_block as rcb  # noqa: E402
from analysis.multisample import run_condition_block as ms  # noqa: E402

if __name__ == "__main__":
    rcb._batched_hybrid_outputs_for_queries = _batched_hybrid_outputs_bennett
    ms.main()
