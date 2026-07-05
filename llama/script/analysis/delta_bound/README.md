# Better delta: approximating the per-block score deviation

The condition needs, per block C and query q,
`delta_C >= max_{i in C} |q (k_i - k_bar_C)| / sqrt(d_k)`.
Production approximates it with the coordinate-wise box (`k_max`/`k_min`, 2 vectors):
efficient but loose — it is the source of the measured `S_delta ~ 1e4–1e7` slack, and
it saturates `tanh(delta/2) ~ 1`, so the value term loses all per-block discrimination.
The sigma (Bennett) route is parked because true sigma is unobtainable at decode; delta
attacks the same slack from the geometry side, and any *over*-estimate of delta keeps
the bound valid.

## Framing

`delta_C(q)` is the support function of the symmetrized convex hull K of the centered
block keys, evaluated at q. Approximating delta with per-block summaries = enclosing K
in a simpler body. In the worst case this is a known hard problem (John's theorem:
ellipsoids are sqrt(d)-tight, and beating them uniformly needs exponentially many
facets), so no candidate can win on all geometries. The bet is that key blocks are
benign: contiguous positions with smooth (RoPE) drift, so K is nearly low-dimensional.

## Candidates (all valid upper bounds unless marked score-only)

| kind | stored per block | q-time | idea |
|---|---|---|---|
| box (baseline) | 2 vec | O(d) | axis-aligned box corner |
| ball | 1 scalar | O(d) | Cauchy–Schwarz: `\|\|q\|\| r_C / sqrt(d_k)` |
| diag_ell | 1 vec + 1 scalar | O(d) | diagonal ellipsoid (scaled radius): `rho \|\|q * w\|\|`, `w_j = max_i \|d_ij\|` |
| moment | 1 vec + 2 scalars | O(d) | `max^2 <= sum^2`: `delta <= sqrt(\|C\|) sigma_hat` (reuses `D_C`, lambda from stage-2 stats) |
| pca-m | m vec + m+1 scalars | O(md) | top-m PCA directions + per-direction extents + residual ball |
| calibrated score | any of the above | O(d) | `c *` estimate with c fit offline (score-only, no guarantee) |

Not pursued yet: sub-block boxes (2x IO), per-coordinate quantiles (score-only),
RoPE-analytic drift bound (needs pre-RoPE keys).

## Plan

0. ~~`delta_variants.py`~~ — done, see Results. Gate ("a storable variant clearly
   tighter than box AND selection error <= 0.95x box") FAILED at block 32; partially
   passed at block 10 (scores only, not bounds).
1. Block 16 (smallest kernel-fast-path block): does the `moment_diag` selection gain
   survive? This is the go/no-go for any kernel work.
2. If yes: PPL at matched budget, main setting (Llama-3.1-8B, wikitext n20, block 16)
   via the runner monkeypatch harness (`ppl_bennett_ms.py` pattern).
3. If yes: kernel IO — `moment_diag` needs only `k_bar` + `D_C` (2 vectors, one less
   than production); stats-kernel microbench (`bennett_kernel_bench.py` pattern).

## Results — stage 0 (2026-07-05)

Llama-2-7b wikitext, layers 10/15/20, 64 groups/layer. Ratio = median
`delta_hat/delta_oracle` over blocks with oracle delta > 0.01; selection = matched-k
true hybrid error vs the box baseline at fractions 5/10/20%. Logs:
`delta_variants.py --budget 0.03125` (block 32) and `--budget 0.1` (block 10).

**Block 32 (generation setting): the delta route hits the same wall as sigma.**

- Oracle median delta is only 2.9–3.4, and un-saturates tanh (sat 0.10–0.14). The box
  is 4.9–5.3x loose (additive gap ~12–13 — this is exactly the `e^13` term-1 slack).
- The hull is high-rank: ball / diag_ell / pca1–4 all land at 4.3–5.6x; pca4 (more IO
  than box) is only ~10% tighter than box. The covariance spectrum is heavy-tailed:
  eig2/eig4 (top-r eigen sketch of `q Sigma q^T`) give 5.1–7.5x because the
  `lam_{r+1} ||q_perp||^2` remainder dominates — same failure as diag+lambda
  (`moment`: 9–11x, selection 1.6–6.8x).
- The only tight storable quantity is the `moment_diag` *score* (`sqrt(|C|) sigma_diag`,
  no remainder, NOT a bound): 1.9–2.1x vs the 2.4x oracle-moment ceiling — but its
  selection is ~neutral (0.89–1.27x), matching the stage-2 sigma finding.
- `box_cal` (box/5): ratio ~1 and tanh un-saturated by construction, yet selection is
  much WORSE (1.5–3.1x). Un-saturating tanh exposes the box's per-block noise to the
  ranking; saturated tanh degrades gracefully to ranking by `p_hat B_C`.
- The oracle-delta ceiling itself is modest: 0.62–0.97x. Nothing storable reaches it.

**Block 10: real headroom, but only for scores, not bounds.**

- Oracle-delta selection: 0.35–0.86x (layer 20: 0.35–0.54x). `moment_orc`
  (`sqrt(|C|) sigma`, oracle) tracks it closely — variance info is ~sufficient.
- `moment_diag` is 1.3–1.6x tight with tanh legitimately un-saturated (sat 0.02–0.04)
  and improves selection on average (~0.67–0.78x at layer 20; ~0.85–1.1x elsewhere).
  pca4/eig4 also help at layer 20 (0.44–0.71x) but are erratic across layers.
- Valid *bounds* stay >= 3.6x loose (best: eig4 at 3.6–4.7x vs box 4.6–5.4x).

**Verdict.** Better storable delta is effectively open in the regime that matters:
worst-case hardness is real and the measured geometry is adversarial (high-rank hull,
heavy-tailed spectrum), so every certified bound stays ~4–5x loose and the certificate
is unchanged. The wins mirror Bennett exactly: small blocks, selection scores rather
than guarantees. The one deployable candidate is `moment_diag` as the selection score —
it stores `D_C` (1 vector) instead of `k_max/k_min` (2 vectors), so it also cuts stats
IO. Next decision point is stage 1 (block 16).
