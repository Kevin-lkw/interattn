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
| `bennett_condition.py` | Stage 1: Bennett vs cosh — slack, Spearman, matched-budget selection, eps->budget calibration |
| `ppl_bennett.py` | PPL-vs-budget sweep with the Bennett condition (wraps runner_cond_block) |
| `bennett_kernel_bench.py` | Stats kernel with Bennett summaries (2 vec + 2 scalars) vs production (3 vec) |

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

## Stage-1 results: tighter / faster / better (2026-07-04)

Scripts: `bennett_condition.py` (offline, oracle sigma), `ppl_bennett.py` (PPL sweep,
monkeypatched runner_cond_block), `bennett_kernel_bench.py` (stats-kernel microbench).

**Tighter — per-cluster yes, certified-eps no (saturation).** G is ~100x smaller than
cosh per cluster and Spearman improves (+0.01–0.02, e.g. 0.9167 -> 0.9399 layer 15).
But the *normalized total bound* and the post-selection certificate are unchanged at
5–20% budgets: once `S_F >> 1` the mass term saturates at `2B` for either F, and the
rest is the (unchanged) tanh value term. Bennett only shows in the certificate when
selection drives `T = sum_unsel p_hat (G-1)` below O(1).

**Better selection at matched budget (offline).** Ranking by the Bennett share beats the
cosh share everywhere: true hybrid error ratios 0.50–0.90 across layers 10/15/20,
block 10/20, frac 5/10/20% (best: layer 20 block 10 frac 5%: 0.504).

**Better PPL at same budget.** Full-model sweep (Llama-2-7b, wikitext_0, block 20,
oracle sigma; cosh baseline = saved `condition_block_runner/budget=0.05`):

| budget (causal) | cosh PPL | bennett PPL |
|---:|---:|---:|
| 0.82 | 3.960 (eps .05) | 3.950 (eps .05) |
| 0.69 / 0.72 | 3.952 (eps .1) | 3.948 (eps .1) |
| 0.34 / 0.38 | 6.330 (eps .5) | 3.984 (eps .5) |
| 0.22 / 0.27 | 13.450 (eps 1) | 4.144 (eps 1) |
| 0.08 / 0.16 | 211.3 (eps 5) | 8.964 (eps 5) |

teacher 3.951. At matched budget ~0.27 the cosh curve interpolates to PPL ~9.7 vs
bennett 4.14 (excess over teacher ~30x smaller). Below budget ~0.15 both collapse
(bennett cliff at eps 10: 1990 @ 0.138; cosh 3766 @ 0.055).

**Faster.** Bennett summaries need `k_bar` + diag-variance + 2 scalars instead of
`k_bar/k_max/k_min` (3 vectors). Stats-kernel microbench (cold L2, production configs,
includes the extra G-cache store): 1.25x @32K, 1.35x @64K, 1.28x @128K.

Caveat: PPL sweep uses oracle `sigma^2(q)` (exact within-block score variance — free in
the eager runner). The kernel path needs the storable bound
`sigma_hat^2 = (sum_j q_j^2 D_j + lambda ||q||^2)/d_k`; quantifying that gap is stage 2.

## Plan

1. ~~Oracle Bennett, offline~~ — done, see above (certificate criterion failed by
   saturation, selection criterion passed strongly).
2. **Verifiable statistics** — `sigma_hat^2` via diag + spectral remainder; scalar
   radius for delta. Re-run `bennett_condition.py` + `ppl_bennett.py` with the storable
   stats; accept if PPL-at-matched-budget keeps most of the oracle gain.
3. **Selection rule decision** — done in effect: Bennett share dominates; keep the
   normalized share form (with G).
4. **Kernel integration** — swap summaries (`D_C`, radius, lambda for `k_max/k_min`),
   fused-path parity sanity (mask/token), LongBench re-validation at matched selected
   ratio, decode ms/step (microbench predicts 1.25–1.35x on the stats kernel).

Efficiency note (per project direction): stages 1–3 ignore implementation efficiency —
they only require that every quantity is *in principle* a per-block prompt-side summary
plus O(1) query-time work, which `D_C`/`lambda_C`/radius all are.
