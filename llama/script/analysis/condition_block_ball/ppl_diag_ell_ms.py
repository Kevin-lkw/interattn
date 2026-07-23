"""Multisample PPL sweep with the diagonal-ellipsoid condition (strict bound).

Same monkeypatch pattern as `ppl_ball_ms.py`, using the diag_ell delta:
per block it stores one deviation-scale vector `w` plus one scalar `rho`
(2 vectors + 1 scalar of summary reads vs the box's 3 vectors), and unlike the
pure ball it keeps per-dimension anisotropy and query-direction dependence.
Pass a dedicated --output-root.
"""

import functools

from ..condition_block_ppl import runner_cond_block as rcb
from ..condition_block_ppl.multisample import run_condition_block as ms
from .ppl_condition import _batched_hybrid_outputs_ball

if __name__ == "__main__":
    print("[diag_ell] using diagonal-ellipsoid delta; --delta-mode is ignored")
    rcb._batched_hybrid_outputs_for_queries = functools.partial(
        _batched_hybrid_outputs_ball, delta_variant="diag_ell"
    )
    ms.main()
