# Additive per-cluster condition

This folder evaluates the simplified condition

```text
g_C = p_hat_C [2 B (cosh(delta_C) - 1)
                 + 2 B_C tanh(delta_C / 2)].
```

For any set `U` of approximate clusters, this is an additive certificate:

```text
||o_hybrid - o|| <= sum_{C in U} g_C.
```

It follows from the tighter set-wise certificate with
`T = sum_{C in U} p_hat_C (cosh(delta_C)-1)` and
`2BT/(1+T) <= 2BT`. The outer `p_hat_C` is part of the formula.

## Question

Does removing the original mass normalizer

```text
S_delta = sum_C p_hat_C cosh(delta_C)
```

produce a useful selection condition, in addition to producing a valid additive
certificate?

The original and simplified mass terms obey

```text
per_cluster_mass_C = S_delta * original_mass_C.
```

Therefore the change is not a harmless algebraic simplification: it multiplies the
mass channel relative to the value channel by a query-dependent `S_delta`. Previous
box-delta measurements put `S_delta` around `1e4-1e7`, so the simplified score may
collapse to term1-only ranking and may be difficult to calibrate with one threshold
across layers and queries.

## Plan

1. [x] Formula checks:
   - verify `per_cluster_mass = S_delta * original_mass`;
   - verify the additive certificate is no smaller than the tight hybrid certificate;
   - rerun random Monte Carlo and real-model violation checks.
2. [x] Offline matched-k selection:
   - Llama-2-7b, WikiText, layers 10/15/20;
   - block sizes 10/16/32 and exact-block fractions 5/10/20%;
   - compare `original`, `per_cluster`, `term1`, and `mass_exp`;
   - report pre/post-`W_o` hybrid error, original top-k overlap, term2 score fraction,
     and certificate slack.
3. [x] Multi-window gate:
   - repeat block 16 on starts 0/1024/2048/3072/4096;
   - require no material matched-k regression before PPL.
4. [x] End-to-end gate:
   - aligned Llama-3.1-8B PPL, block 16, first two layers dense;
   - compare against the saved original-condition curve at matched measured budgets;
   - interpolate NLL per window, never PPL directly.

## Commands

```bash
cd llama
python -m script.analysis.condition_block_ppl.per_cluster_condition.offline \
  --device cuda:0 --block-size 16 --layers 10 15 20

python -m script.analysis.condition_block_ppl.per_cluster_condition.ppl \
  --model meta-llama/Llama-3.1-8B --device cuda:0 --dtype bfloat16 \
  --ppl-protocol aligned --seq-len 1024 --num-samples 5 \
  --start-offset 5120 --sample-stride 1024 --block-size 16 \
  --eps 1e5 3e5 1e6 3e6 1e7 3e7 1e8 3e8 1e9 3e9 1e10 \
  --output-root <output-root>
```

Offline outputs default to
`llama/result/per_cluster_condition/<model>/<dataset_start>/block_<size>/`.

## Results

### 1. Formula and certificate checks

- Unit tests pass for both identities: the simplified mass term is exactly
  `S_delta` times the original mass term, and the additive certificate is no
  smaller than the tighter set-wise certificate.
- The existing random checks still pass. The worst observed error/bound ratios
  were `1.000` for TV, `1.000` for Bennett, `0.898` for the all-approx bound,
  and `0.752` for the hybrid bound.
- On real Llama-2 activations, the additive certificate had `0/5760` violations
  in the five-window block-16 experiment (`box` and `oracle` delta combined).

Conclusion: this is a valid additive certificate for a selected set of
approximate clusters. The experiment does not establish that each `g_C` is a
standalone bound on the error caused by cluster `C` in isolation.

### 2. Why the simplified score saturates

For block 16 and box delta, averaged over five windows and layers 10/15/20:

- mean `log10(S_delta) = 5.894`;
- the value-channel term contributes only `9.0e-7` of the simplified score;
- `per_cluster` and `term1` therefore select the same clusters at all reported
  fractions to displayed precision.

Removing `S_delta` does not merely loosen the bound. It changes

```text
original = term1 / S_delta + term2
```

into

```text
per_cluster = term1 + term2 = S_delta * (term1 / S_delta) + term2.
```

Thus the mass channel is amplified by about `1e6` relative to the value channel.
With oracle delta, mean `log10(S_delta)` falls to `1.680` and the value-channel
fraction rises to `0.0112`, so the collapse is less complete. Oracle delta is a
diagnostic and is unavailable to the deployed selector.

### 3. Offline matched-k selection

Block 16, five WikiText windows, Llama-2-7b, layers 10/15/20. Values below are
post-`W_o` hybrid error divided by the original-condition error. Confidence
intervals are Student-t intervals over windows.

| delta | exact fraction | per-cluster/original | 95% CI | overlap with original |
|---|---:|---:|---:|---:|
| box | 0.05 | 0.944 | [0.858, 1.029] | 0.816 |
| box | 0.10 | 1.077 | [0.950, 1.204] | 0.795 |
| box | 0.20 | 1.187 | [1.065, 1.309] | 0.813 |
| oracle | 0.05 | 0.960 | [0.944, 0.976] | 0.845 |
| oracle | 0.10 | 0.985 | [0.968, 1.002] | 0.835 |
| oracle | 0.20 | 0.955 | [0.928, 0.982] | 0.842 |

The deployable box-delta score is significantly worse at 20% exact blocks. A
single-window block-size check shows the same failure mode becoming stronger as
`S_delta` grows: the 20% post-`W_o` ratios are `1.131` at block 10 and `1.425`
at block 32.

Full tables are saved under
`result/per_cluster_condition/Llama-2-7b-hf/`; the five-window aggregate is
`wikitext_multiwindow/block_16/summary.md`.

### 4. Aligned held-out PPL

Setup: Llama-3.1-8B, five disjoint WikiText windows of length 1024 starting at
5120, block 16, first two layers dense, bfloat16. NLL is interpolated separately
within each window at the measured attention budget and then aggregated.

| budget | original PPL | term1 PPL | per-cluster PPL | per-cluster/original |
|---:|---:|---:|---:|---:|
| 0.20 | 14.8628 | 34.5071 | 93.4546 | 6.288x |
| 0.25 | 8.2616 | 17.0825 | 30.4084 | 3.681x |
| 0.30 | 7.1929 | 11.7988 | 14.6712 | 2.040x |
| 0.35 | 7.0911 | 9.5814 | 11.2765 | 1.590x |
| 0.39 | 7.0251 | 8.8767 | 9.8335 | 1.400x |

For every budget, the paired 95% confidence interval for
`NLL(per_cluster) - NLL(original)` is strictly positive. The simplified score is
also worse than term1-only at every matched budget. The PPL curve and matched
comparison are saved under
`result/per_cluster_condition_ppl/Llama-3.1-8B/aligned_heldout_n5_s1024_b16/`.

### Decision

**Reject this modification as a cluster-selection condition.** It is a negative
result rather than a candidate that needs more threshold tuning:

- removing `S_delta` structurally amplifies the mass channel by about `1e6` under
  box delta, so term2 is numerically erased;
- matched-k selection consequently collapses to term1-only, with a significant
  18.7% post-`W_o` regression at the 20% exact-block setting;
- a global threshold makes the result still worse because `S_delta` becomes an
  implicit query-, KV-head-group-, and layer-dependent threshold multiplier;
- held-out matched-budget PPL is worse than both the original condition and
  normalized term1-only at every tested budget.

No additional selector tuning or deployment integration is recommended for this
form. The original normalized condition remains the selector. The formula may be
kept only as a mathematically valid additive certificate computed after selection
when a decomposable reported bound is specifically required; otherwise this
experiment should be treated as closed.
