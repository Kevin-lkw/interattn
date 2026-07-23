# representative_optimality — is avg-K / avg-V the optimal cluster representative?

Explores whether the arithmetic means `k_bar`, `v_bar` are the best query-independent
representatives, under the objective **minimise the true output error** `||o_hat - o||`,
with the bound kept as a valid/computable constraint, over the query distribution that
actually compresses each cluster (query-aware). Oracle `u_C(q)` is the ceiling. Imports
`query_cond_dist` / `per_cluster_condition`; modifies no existing code.

## Theory (`derivation.tex`)

- **Prop 1 (value):** the storable minimiser of `E_q||v* - u_C||^2` is the conditional
  mean `v* = E_q[u_C] = sum_i wbar_i v_i` (`wbar_i = E_q[w_i]`), still one vector. `v_bar`
  is optimal iff `wbar` is uniform. `v_bar` minimises the worst-case certificate
  (Chebyshev centre → `2 B_C tanh(delta/2)`); `v*` minimises the actual error. This is the
  precise sense in which "the bound changes off avg": worst-case-optimal vs average-optimal.
- **Prop 2 (key):** any `k* = k_bar + Delta` multiplies the mass estimate by `e^{q.Delta}`
  and destroys the mean-zero hypothesis behind `mean(e^x) <= cosh delta`, so the mass
  certificate loosens; a shifted key can reduce the *average* mass bias but not the
  per-query one.
- **Prop 3 (coupling):** mass and value channels share the normaliser and anti-correlate,
  so they can't be optimised independently; optimise `v*` at fixed `k_bar` first.

## Stage 2 — offline measurement (`offline.py`)

At a per-query budget `b` the exact set is chosen by the real condition; compressed
clusters use a representative; we measure the true `||o_hat - o||/||o||` via the identity
`sum_{i in C} e^{s_i} v_i = Z_C u_C`. **Calibration/eval split** (even queries build
representatives, odd queries evaluate) removes in-sample optimism — essential, it flips
the key-representative conclusion.

Variants — value (mass fixed `k_bar`): `vbar` (baseline), `vnorm`, `v_all` (attention-
weighted over all cal queries), `v_qw` (attention-weighted over compressing cal queries),
`oracle_u` (`u_C(q)`, value ceiling). Key (value `v_bar`): `k_qw` (`sum_i wbar_i k_i`),
`both_qw`; `oracle_m` (true mass, mass ceiling).

```
python -m ...representative_optimality.offline --layer 15 --budget 0.1 \
  --select-budgets 0.0 0.1 0.2 --sample-heads 8 --device cuda:2
```

### Findings (Llama-2-7b, block 10, wikitext, 8 heads, held-out eval)

Relative true error `||o_hat-o||/||o||` at layer 15:

| budget | vbar | v_all | v_qw | oracle_u | k_qw | both_qw | oracle_m |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1113 | 0.1052 | 0.1052 | 0.0965 | 0.1370 | 0.2522 | 0.1040 |
| 0.2 | 0.0554 | 0.0532 | 0.0532 | 0.0476 | 0.0554 | 0.0654 | 0.0474 |

- **avg V is near-optimal, not optimal.** The attention-weighted value centroid beats
  `v_bar` by ~4-6% and generalises, closing ~40% of the `v_bar → u_C` value-oracle gap.
  Modest, one stored vector, no bound change (Prop 1 / Cor 1).
- **query-aware buys nothing over all-query:** `v_qw ≈ v_all` — the within-cluster weight
  `wbar` is essentially query-independent, so Prediction 2 fails.
- **avg K is effectively optimal for deployment.** `k_qw` is worse than `v_bar` at
  budgets 0.1/0.2 and breaks the bound; its only gain is in pure approximation (`b=0`,
  where it corrects the Jensen mass under-estimate) — not the deployment regime.
- (`b=0` pure-approximation is a mass-dominated regime with ~300% error and is reported
  only for the decomposition, not as a usable setting.)

### Cross-layer consistency (Llama-2-7b, budget 0.1, held-out eval)

| layer | vbar | v_all=v_qw | oracle_u | v gain | value-gap closed | k_qw |
|---|---:|---:|---:|---:|---:|---:|
| 5  | 0.0818 | 0.0721 | 0.0634 | -11.9% | 53% | 0.0852 |
| 10 | 0.1856 | 0.1659 | 0.1440 | -10.6% | 47% | 0.1810 |
| 15 | 0.1113 | 0.1052 | 0.0965 |  -5.5% | 41% | 0.1370 |
| 20 | 0.1629 | 0.1602 | 0.1495 |  -1.7% | 20% | 0.2537 |
| 27 | 0.1367 | 0.1266 | 0.1185 |  -7.4% | 55% | 0.5038 |

- The value gain (attention-weighted centroid over `v_bar`) is **positive at every layer**
  (-2% to -12%, closing 20-55% of the value-oracle gap) and generalises to held-out
  queries. `v_qw ≈ v_all` at every layer — query-aware weighting gives nothing beyond a
  plain attention-weighted mean.
- `k_qw` is worse than `v_bar` at every layer (mildly early, badly late: 0.50 vs 0.14 at
  L27) — avg K is effectively optimal; `both_qw` only wins at L5 (coupling, Prop 3).
- The value channel generally has more headroom than mass (`oracle_u < oracle_m`), but the
  storable `v*` captures only ~1/2 of it — the rest is per-query (`u_C(q)`), not storable.

A cheap **storable proxy** recovers the whole gain: `v_qmean = sum_i softmax(q_bar.k_i) v_i`
with `q_bar` the mean calibration-query direction (one vector per head) matches `v_all`
at every layer (e.g. L10: 0.1652 vs 0.1659). This makes `v*` deployable and is what
Stage 4 uses.

**Verdict:** avg V is near-optimal, not optimal — an attention-weighted value centroid is a
small, consistent, bound-preserving improvement (one stored vector); avg K is effectively
optimal for deployment.

Per-layer TSVs: `result/representative_optimality/<model>/layer<L>_block<B>/offline.tsv`.

## Stage 4 — matched-budget PPL (`ppl.py`) — PASS

Paired PPL: at each eps the model runs twice (stock `v_bar` vs a monkeypatched hybrid
using `v* = sum_t softmax(q_bar.k_t) v_t`, `q_bar` = mean query direction of the head over
the window, within-cluster weights over visible tokens only → leakage-free). Selection is
independent of the value representative, so both runs share the exact same budget — a clean
paired matched-budget test. Only `runner_cond_block._batched_hybrid_outputs_for_queries` is
monkeypatched; no existing file is edited.

```
python -m ...representative_optimality.ppl --seq-len 1024 --num-samples 4 \
  --start-offset 2048 --sample-stride 1024 --block-size 10 --eps 8 4 2 1 --device cuda:2
```

Llama-2-7b, block 10, wikitext aligned, 4 windows, teacher PPL 8.30:

| eps | budget | base (v_bar) | v* | ppl_ratio | student-teacher gap closed |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.169 | 1018.5 | 601.5 | 0.591 | (collapse regime, both unusable) |
| 4 | 0.184 |  26.66 | 21.09 | 0.791 | (collapse regime) |
| 2 | 0.198 |   8.86 |  8.64 | 0.975 | **39%** |
| 1 | 0.217 |   8.46 |  8.40 | 0.993 | **37%** |

`v*` lowers PPL at **every eps and every one of the 4 windows** (all `dNLL < 0`). In the
usable regime (PPL near teacher, budget ~0.2) it consistently closes ~37-39% of the
student-teacher gap; at aggressive budgets it partially rescues the collapse (ratio
0.59-0.79). The single-layer ~5% error gain compounds across layers into a stable PPL win.
`v*` is one stored vector, cheap (one `q_bar` per head, available at prefill), and preserves
the certificate. Caveat: `q_bar` here is taken from the eval window itself (a low-capacity
1-vector/head statistic whose generalisation the offline held-out split already confirmed);
a fully held-out calibration of `q_bar` is the natural follow-up before deployment.
