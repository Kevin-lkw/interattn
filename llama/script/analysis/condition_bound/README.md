# Condition-bound theory: verification, dead ends, and the variance (Bennett) upgrade

Theory-side experiments for the condition-block error bound. Code here answers:
is the bound correct, where is it loose, which refinements work, and what is the
next update. Setup for all GPU scripts: Llama-2-7b-hf, wikitext, seq_len 1024,
4 heads x 16 queries per layer (layers 10/15/20), block_size = round(1/budget) = 10 —
same as `result/Llama-2-7b-hf/wikitext_0/condition_block_corr/`.

## Files

| file | what it shows |
|---|---|
| `derivation.tex` | New bound: variance-constrained (Bennett) replacement of `cosh(delta)`, hybrid per-step certificate, per-cluster guarantee |
| `sanity_mc.py` | CPU Monte Carlo for every inequality (TV lemma, Bennett lemma, all-approx bound, hybrid certificate) — all PASS |
| `bound_variants.py` | Bound validity/slack + Spearman for B_C / global-B variants |
| `error_decompose.py` | True error split into mass vs value channels; `b_C ~ Var(s)/2` fit |
| `hybrid_guarantee.py` | Per-cluster guarantee, hybrid certificate validity, certified-score vs share-rule ranking |
| `estimator_correction.py` | Estimator-correction ablation (negative result) |
| `selection_kernel_bench.py` | Cost of the S_delta normalizer in the Triton stats kernel (~0.1%, free) |

Run from repo root, e.g. `python llama/script/analysis/condition_bound/hybrid_guarantee.py --device cuda:0`
(nanogpt env; flags: `--device`, `--budget`, `--layers`).

## Verified facts

- **The bound holds.** Summed per (head, query) as the theorem states: 0 violations in
  192 groups; median error/bound 5–11% (~10–20x conservative). 50k-instance Monte Carlo:
  max ratio 0.86. The per-cluster *share* is a diagnostic, not a guarantee (~0.3% rows
  exceed it); the literal per-cluster guarantee
  `p_hat[B max(cosh d - 1, (S-1)/S) + 2 B_C tanh(d/2)]` has 0/4128 violations but is
  ~S_delta/2 worse in sum (no cancellation) — use it per-cluster only.
- **Hybrid certificate.** For the production computation (selected blocks exact inside the
  shared softmax), with `T = sum_unsel p_hat (cosh d - 1)`:
  `err <= 2B T/(1+T) + sum_unsel 2 p_hat B_C tanh(d/2)` — valid for any selection, with
  pure-approx `p_hat` (0 violations at all tested fractions + 30k MC). Additive per
  cluster, no normalizer, computable at decode time as a per-step certified epsilon.
- **Error anatomy.** True error splits ~55% mass / ~45% value; the two channels
  anti-correlate. `b_C ~ Var(s)/2` with corr 0.95 (median ratio 1.05–1.11) while the
  bound charges `cosh(delta_range)` with `S_delta ~ 1e4–1e7` — that is the entire slack.

## Dead ends (measured; do not revisit without a new idea)

- Centered `B_C = max||v_i - v_bar||`: bound shrinks only 5–8%.
- Centered global B via `o_hat` (`b_star`): bound gets ~1.5x *looser*; oracle global
  constant only ~25% tighter — value-side constants are not the slack.
- Un-normalizing term1 (dropping `/S_delta`): kernel saving ~0.1% (normalizer reuses
  loaded data; stats kernel is read-bound), while `S_delta` varies ~200x across queries,
  so a fixed eps then over-selects massively. Keep the normalizer for selection.
- Certified (unnormalized) score as the *ranking*: 3–52% higher true error than the
  production share rule at matched budget (cosh overweights the mass channel).
- Estimator corrections: oracle `+Var(s)/2` mass fix cuts mass error to 18–30% but
  *worsens* total error (109–139%); diag-cov version ~neutral; first-order value fix
  (`v_bar + Cov(v,s)`, exact scores) also hurts (102–108%); joint achievable-order
  93–129%. Only the joint oracle (exact mass + exact `u_C`) reaches 15–25% — headroom
  exists but no storable summary reaches it. The plain mean estimator is
  self-compensating: keep it.

## The update: variance-constrained (Bennett) first term

See `derivation.tex`. One inequality changes: `e^{b_C} <= cosh(delta_C)` becomes

```
e^{b_C} <= G(sigma_C, delta_C) = (sigma_C^2 e^{delta_C} + delta_C^2 e^{-sigma_C^2/delta_C}) / (sigma_C^2 + delta_C^2)
```

with `sigma_C^2 = q Sigma_C q^T / d_k` (within-cluster key covariance). `G(d,d) = cosh(d)`
(strict generalization), tight, one-sided, safe with over-estimates of sigma/delta.
At observed values (`sigma ~ 1`, `delta_range ~ 10–17`) term1 tightens ~50–150x.
Verifiable sigma bound: store `diag(Sigma_C)` (1 vector) + off-diagonal spectral norm
(1 scalar) per block — replaces `k_max/k_min` (2 vectors), so the stats kernel reads
*less* than today. Expected wins: certificate ~2 orders tighter; certified score may
become a usable ranking (its failure was traced to cosh overweighting); fewer summary
bytes in the kernel that grows with context.

## Plan

1. **Oracle Bennett, offline** — add `G` to this harness with exact `sigma_C^2(q)`,
   range-bound delta. Verify: 0 violations; post-selection certificate slack improves
   >= 30x; Spearman >= 0.88 baseline; matched-budget ranking with Bennett score vs
   share rule.
2. **Verifiable statistics** — `sigma_hat^2 = (sum_j q_j^2 D_j + lambda ||q||^2)/d_k`
   (diag + spectral remainder, prefill-computed); optionally scalar radius for delta.
   Accept if degradation vs oracle sigma < 2x.
3. **Selection rule decision** — Bennett-scored condition vs production share at matched
   budget; pick by true error, spot-check LongBench.
4. **Kernel integration** — replace `k_max/k_min` reads with `D_C` + 2 scalars in the
   selection-stats kernel (bytes/block: 3 vectors -> 2 vectors + 2 scalars); sanity:
   mask parity vs eager reference, token parity e2e; re-validate LongBench at matched
   selected ratio; measure decode ms/step (stats kernel is the context-growing cost).

Efficiency note (per project direction): stages 1–3 ignore implementation efficiency —
they only require that every quantity is *in principle* a per-block prompt-side summary
plus O(1) query-time work, which `D_C`/`lambda_C`/radius all are.
