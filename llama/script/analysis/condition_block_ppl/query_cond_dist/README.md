# query_cond_dist — which queries a cluster can compress

For a fixed contiguous block cluster `C` (members `[root, root+B)`, so `k_bar_C`
and the key spread are constant across queries), sweep every causal query `q`
that fully sees `C` and evaluate the same coupled selection condition used in
`condition.py`:

```
s_C     = q . k_bar_C / sqrt(d)          # = segment mean of token scores
delta_C = max_i |q.(k_i - k_bar_C)/sqrt(d)|
p_C     = softmax_C(log|C| + s_C)
cond_C  = p_C (2B (cosh delta_C - 1) / sum_C' p_C' cosh delta_C'  + 2 B_C tanh(delta_C/2))
```

`C` is **compressed** for `q` when `cond_C <= eps`. The script groups the
`(q, cond_C)` pairs by cluster and looks at the distribution — especially the
**mean** — of the queries that get compressed, relative to the cluster geometry
(`k_bar_C`, top centered-key direction `u1`) and to the overall query mean.

## Run

```
conda run -n nanogpt python -m llama.script.analysis.condition_block_ppl.query_cond_dist.run \
  --layer 15 --budget 0.1 --seq-len 2048 --query-start 256 --num-queries 256 \
  --sample-heads 4 --condition-eps 0.01 --device cuda:2
```

`--budget` sets `block_size = round(1/budget)`; `--condition-eps` sets the
compression threshold. Outputs go to `result/query_cond_dist/<model>/layer<L>_block<B>_eps<eps>/`:
`cluster_query_dist.tsv` (per-cluster stats), `aggregate.png`, `example_clusters.png`.

## Implementation

The per-query condition is vectorised via the reshape identity `s_C = mean of
in-block token scores` (blocks are contiguous, fixed size), reusing
`per_cluster_condition.scores.condition_scores`. Startup runs a sanity check
against the reference `condition._cluster_condition_rows_and_sanity` (max rel
error ~2e-6). The partial current block is kept in the softmax / `B_all` for
faithfulness but excluded from the per-cluster analysis (only full blocks).

`q` is post-RoPE (the space the condition operates in), so vector means mix RoPE
phases; per-query scalars (`s`, `cos`, `delta`) are phase-consistent.

## Finding (Llama-2-7b, layer 15, block 10, wikitext, 4 heads, 729 clusters)

The queries a cluster compresses are the ones that **point away from the
cluster's mean key** — i.e. low / negative `q·k_bar`, so the cluster carries
negligible softmax mass `p_C` for that query. Compression happens on the
low-attention tail, exactly where the mean representative is safe.

| eps | frac compressed | cos(q_bar,k_bar) comp / exp | mean s comp / exp | mean delta comp / exp | mean \|q\| comp / exp |
|----|----|----|----|----|----|
| 1.0  | 0.978 | -0.47 / -0.08 | -8.19 / -2.29 | 2.44 / 2.98 | 15.31 / 15.42 |
| 0.1  | 0.848 | -0.49 / -0.23 | -8.64 / -4.18 | 2.43 / 2.65 | 15.36 / 14.85 |
| 0.01 | 0.572 | -0.55 / -0.35 | -9.85 / -6.11 | 2.40 / 2.40 | 15.58 / 15.00 |

- **Mass, not spread, drives selection.** `s = q·k_bar/sqrt(d)` separates
  compressed from expanded at every cluster (all points below the diagonal in
  `aggregate.png`; `cos(q_bar,k_bar)` histograms are cleanly bimodal). `delta_C`
  (key spread) barely differs and the gap vanishes as `eps` tightens.
- **Direction, not norm.** `|q|` is ~equal for both sets; the split is angular.
- **A distinct region.** In per-cluster q-PCA (`example_clusters.png`) the
  compressed queries occupy one lobe and their mean (star) is well separated
  from the expanded-query mean. `cos(q_bar_comp, q_bar_all)` drops 1.00 -> 0.97
  and `|q_bar_comp|/|q_bar_all|` rises 1.00 -> 1.05 as eps tightens: the
  compressed subset's mean shifts modestly into the anti-`k_bar` direction.
- 100% of clusters have `cos(comp,k_bar) < cos(exp,k_bar)` at every eps.

## Consistency sweep (`sweep.py`)

`sweep.py` loads the model once and re-runs the summary across layers / blocks /
heads; run it per data sample or model. The headline result — compressed queries
sit at more negative `cos(q,k_bar)` and more negative `s`, with `delta` a weak
discriminator and `|q|` none — holds everywhere tested (`--condition-eps 0.1`,
8 heads, 256 queries, seq 2048):

```
python -m ...query_cond_dist.sweep --layer 15 --layers 2 6 10 14 18 22 26 30 --budget 0.1 --device cuda:2
```

- **Depth** (Llama-2, layers 2..30): `frac_lower = 1.00` at every layer; `cos`
  and `s` lines cleanly separated (compressed below expanded), per-head dots
  keep the separation (`layers.png`). `delta` lines nearly touch; `|q|` lines
  overlap. `frac_comp` drifts 0.98 (L2) -> 0.74 (L30).
- **Model / GQA** (Llama-3.1-8B, layers 2..30): identical pattern, `frac_lower =
  1.00` everywhere. GQA needs Q->KV head mapping (`gather_head_qkv`); with 32 Q /
  8 KV heads `q_head // 4` selects the KV head.
- **Block size** (5 / 10 / 20 at layers 10/15/20): `frac_lower = 1.00` for all;
  `cos`/`s` gap persists; `delta` gap stays small. `frac_comp` falls with larger
  blocks (e.g. L15: 0.93 @ B5 -> 0.66 @ B20).
- **Data sample** (wikitext start 0/1/2/3, layers 10/15/20): numbers identical to
  ~3 decimals (e.g. L15 `s` comp/exp = -8.62/-4.59, -8.60/-4.61, -8.62/-4.64,
  -8.59/-4.60) — the geometry is a property of the model, not the sample.
- **eps** (1.0 / 0.1 / 0.01, see table above): the `cos`/`s` split sharpens as
  eps tightens; `delta` gap shrinks to ~0. Mass, not spread, is the selector.

## Boundary check — why "compress low q.k_bar" is not the selection rule (`boundary.py`)

The marginal finding above (compressed clusters have small `q.k_bar`) does NOT
mean the selector is a threshold on `q.k_bar`: `term_ablation` showed the `p_hat`
control (rank by `q.k_bar`) is 1.8-3.1x worse matched-k error. `boundary.py`
resolves the apparent contradiction by running the full-condition selector vs a
mass-only selector at a matched per-query budget and cross-tabulating decisions
in `(s_rel, delta)` where `s_rel = s - per-query mass threshold`.

The compress/expand boundary is **tilted, not vertical** — it is `s + log f(delta)
= const`, so higher `delta` moves the expand region to lower `s`. At budget 0.2
(Llama-2 L15, 235k pairs, 8 heads): ~8.7% of decisions differ from mass-only
(Jaccard 0.64 on the expand set), split into exactly the two predicted wedges:
- full **compresses** despite high `q.k_bar` (`s_rel>0`): mean `delta = 1.71`
  (low) — the "large q.k_bar but small delta, so approximate" set (4.7% of all
  compressed).
- full **expands** despite low `q.k_bar` (`s_rel<0`): mean `delta = 5.68` (high).

The `P(expand)` heatmap (`boundary_b0.2.png`) shows the diagonal boundary directly.
The ~9% flipped decisions carry most of the error because the high-delta clusters
mass-only wrongly compresses are exactly where the mean representative is worst
(`cosh(5.68)` is huge) — a small, high-leverage set. This is the geometric reason
`delta` cannot be dropped, matching `term_ablation` (`p_hat` Spearman(gain)~0) and
the diag_ell / ball-bound results.

```
python -m ...query_cond_dist.boundary --layer 15 --budget 0.1 --budgets 0.1 0.2 --device cuda:2
```
