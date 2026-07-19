# Term 1 / Term 2 ablation

This folder answers one decision question: which part of the condition score should
drive exact-block selection, and what can be removed without hurting the final hybrid
attention output?

The original per-block score is

```text
term1_C = p_hat_C * 2 B * (cosh(delta_C) - 1) / S
term2_C = p_hat_C * 2 B_C * tanh(delta_C / 2)
full_C  = term1_C + term2_C
S       = sum_C p_hat_C cosh(delta_C)
```

## What we already know

- With box delta, `S` is large (`~1e5` for block 10 and `~1e7` for block 32), so the
  sum of term 1 is effectively fixed at `2B`.
- Box delta saturates `tanh(delta/2)` for more than 99% of blocks, reducing term 2 to
  a ranking close to `p_hat_C B_C`.
- Term 2 has the best Spearman correlation with local block error, but term2-only
  selection has produced much worse matched-k hybrid error.
- Term 1 is concentrated in the selected tail. Term1-only ranking is close to the full
  rule, and the improvement from oracle delta primarily enters through term 1.
- Therefore Spearman against `||o_C - o_hat_C||` is a diagnostic, not the final
  selection objective. Vector cancellation and the shared softmax normalizer matter.

## Questions and hypotheses

1. Does term 1 or term 2 reduce the final hybrid error at a fixed number of exact
   blocks? Hypothesis: term 1 carries most or all useful selection signal.
2. Why does term 2 look good under Spearman? Hypothesis: it predicts bulk local error,
   but not the marginal reduction in the vector-valued final error.
3. Can the score be simplified? Hypotheses:
   - For term1-only ranking, `B` and `S` can be removed exactly because they are
     query-level constants.
   - `term1_log_exact = log(p_hat) + log(cosh(delta)-1)` has exactly the same ordering
     as term 1 and avoids computing `B` and `S`.
   - With box delta, `term1_log_exp = log(p_hat) + delta` should be nearly equivalent.
   - The `-1` and `tanh` can be dropped inside the full score with little change under
     box delta.
   - A coefficient sweep should determine whether term 2 should be downweighted or
     removed; its high local correlation alone is not a reason to retain weight 1.

## Experiment matrix

Use Llama-2-7b on one WikiText window for the diagnostic stage, matching the previous
setup: layers 10/15/20, four evenly spaced heads, 16 query positions, and exact-block
fractions 5/10/20%.

Run all combinations of:

- Block size: 10, 16, 32.
- Delta: production box delta and oracle exact delta.
- Main scores: `full`, `term1`, `term2`.
- Exact simplification: `term1_log_exact`.
- Deployable approximations: `term1_log_exp`, `term2_sat`, `full_no_minus1`,
  `full_mass_exp`, `full_t2_sat`, and `simple_full`. `full_mass_exp` simplifies only
  the mass channel and keeps the exact low-delta behavior of term 2.
- Low-delta-safe tanh approximations: a singleton-size gate and
  `min(delta/2, 1)`. These test whether the rare unsaturated tail blocks explain the
  regressions from replacing every tanh value by one.
- Term-2 coefficient: `term1 + lambda * term2` for lambda 0.25, 0.5, 1, and 2.
- Combined deployment candidates: replace term 1 by its mass-exponential form and
  use term-2 weights 0.25, 0.5, or 0.75.
- Controls: `p_hat`, `p_hat * B_C`, and `delta` alone.

`simple_full` uses

```text
2 B * softmax_C(log(p_hat_C) + delta_C) + 2 p_hat_C B_C,
```

which removes `cosh`, `-1`, and the explicit `S` calculation.

## Metrics

The script reports separate metrics for separate questions:

1. Local prediction: Spearman, top-k overlap, captured positive relevance, and NDCG
   against the true per-block contribution error.
2. Selection-aware prediction: the same metrics against the exact single-block
   marginal gain, defined as the all-approximate error minus the error after making
   only block C exact.
3. Primary endpoint: the true matched-k hybrid output error after selecting the top-k
   blocks. This includes vector cancellation and the mixed exact/approximate softmax
   normalizer.
4. A stronger local endpoint: the same error after applying that head's column block
   of `W_o`. This measures the actual residual-stream contribution of the head.
5. Optional ceiling: greedy recomputation of marginal gain after every selected block.

Top-k overlap with local error is intentionally not treated as the winner criterion.
The primary comparison is paired matched-k hybrid error; post-`W_o` error breaks close
ties.

## Decision rules

- Effective term: improves mean matched-k error at both 10% and 20% on at least two of
  three layers, without a material post-`W_o` regression.
- Safe simplification: error ratio <= 1.02 versus its unsimplified reference and
  top-k agreement >= 0.95 for box delta. Exact algebraic equivalence is checked by a
  unit test where applicable.
- Drop term 2: term1-only or a coefficient below 0.25 is no worse than full by more
  than 2% on average and wins on the majority of layer/block-size settings.
- Advance to PPL: a rule must improve post-`W_o` error by at least 5% in the block-16
  and block-32 settings. PPL/downstream evaluation is too expensive to use as the
  first screening stage.

## Commands

Run the focused diagnostic:

```bash
cd llama
python -m script.analysis.condition_block_ppl.delta_bound.term_ablation.run \
  --device cuda:0 --block-size 10 --layers 10 15 20
```

Repeat with `--block-size 16` and `--block-size 32`. Add `--greedy-oracle` only for the
final shortlisted settings because it recomputes a marginal choice at every step.

Aggregate independent windows after the per-window runs:

```bash
python -m script.analysis.condition_block_ppl.delta_bound.term_ablation.summarize \
  --block-size 16 --starts 0 1024 2048 3072 4096
```

Run an aligned PPL sweep for one score variant:

```bash
python -m script.analysis.condition_block_ppl.delta_bound.term_ablation.ppl \
  --model meta-llama/Llama-3.1-8B --device cuda:0 --dtype bfloat16 \
  --ppl-protocol aligned --seq-len 1024 --num-samples 5 \
  --start-offset 5120 --sample-stride 1024 --block-size 16 \
  --eps 0.05 0.075 0.1 0.15 0.25 0.5 1 2 4 8 \
  --score-kind mass_exp --term2-weight 0.5 --output-root <output-root>
```

Use `summarize_ppl.py` to compare the saved `summary.pt` files. It linearly
interpolates NLL, not PPL, within each window and reports paired confidence intervals
at the requested measured budgets.

Outputs are written under
`llama/result/term_ablation/<model>/<dataset_start>/block_<size>/` as `summary.json`
and `summary.md`.

## Stages

1. Term screen: block 10 and 32, pre/post-`W_o`, all score variants.
2. Deployment gate: block 16, then repeat across at least five WikiText windows for
   paired confidence intervals.
3. End-to-end gate: matched-budget teacher-student KL/NLL and PPL for only the full
   score and the best one or two simplified rules.
4. Kernel gate: measure summary storage, query-time FLOPs, and selection-kernel time
   only if the simplified rule survives stage 3.

## Results (2026-07-18)

### Stage 1: block 10/16/32 screen

One WikiText window, 3 layers x 64 head-query groups per block size. The table averages
the ratios over all 27 box-delta settings (3 block sizes x 3 layers x 3 fractions).

| score | pre error / full | post-Wo error / full | worst pre ratio | conclusion |
|---|---:|---:|---:|---|
| term1 only | 1.099 | 1.065 | 2.184 | Main signal, but term2 cannot always be deleted |
| term2 only | 3.220 | 3.157 | 5.037 | Strong local correlation, unusable selector |
| full without `-1` | 1.000 | 1.000 | 1.000 | Safe in every tested box setting |
| `full_mass_exp` | 1.000 | 1.000 | 1.000 | Near-identical top-k; removes term1 `cosh`, `-1`, explicit `S` |
| full with every tanh set to 1 | 1.019 | 1.015 | 1.248 | Unsafe because short tail blocks are unsaturated |
| term2 weight 0.25 | 0.949 | 0.941 | 1.075 | Best average candidate |
| term2 weight 0.50 | 0.954 | 0.951 | 1.054 | More conservative candidate |

Term2's within-query Spearman against local block error is typically 0.84-0.96, but
its Spearman against single-block marginal gain is much lower (often near zero), and
term2-only matched-k error is 3-5x worse. Term1 usually has the higher marginal-gain
correlation. This confirms that local-error top-k is not the selection objective.

`full_mass_exp` is

```text
2 B * softmax_C(log(p_hat_C) + delta_C)
+ 2 p_hat_C B_C tanh(delta_C / 2).
```

For box delta its average top-k agreement with the original full score was at least
0.9998, and the error difference rounded to zero. Replacing tanh by one is not safe:
the rare final partial/singleton block can have small delta. A size gate or
`min(delta/2, 1)` fixes block 16/32 in this sample, but still has a 7.7% worst
regression at block 10, so retaining tanh is preferable.

### Stage 2: five-window block-16 gate

The next table reports post-`W_o` error ratios for box delta. Brackets are 95% Student-t
intervals over five independent WikiText windows; each window aggregates layers
10/15/20 and 64 groups per layer.

| score | 5% exact | 10% exact | 20% exact |
|---|---:|---:|---:|
| term1 only | 0.944 [0.858, 1.029] | 1.077 [0.950, 1.204] | 1.187 [1.065, 1.309] |
| term2 only | 2.475 [1.709, 3.241] | 3.638 [2.091, 5.185] | 5.078 [2.650, 7.507] |
| term2 weight 0.25 | 0.922 [0.857, 0.987] | 0.932 [0.875, 0.990] | 0.958 [0.925, 0.992] |
| term2 weight 0.50 | 0.942 [0.904, 0.979] | 0.943 [0.890, 0.996] | 0.965 [0.918, 1.011] |
| `full_mass_exp` | 1.000 | 1.000 | 1.000 |

Combining the mass simplification with weights 0.25/0.50/0.75 produced identical
reported box-delta metrics to the corresponding unsimplified weighted score.
The strongest deployment candidate from this stage is therefore

```text
score_C = 2 B * softmax_C(log(p_hat_C) + delta_C)
        + 0.5 p_hat_C B_C tanh(delta_C / 2),
```

where the second line is `0.25 * term2`. It improves the five-window local endpoint by
about 4-8% while deleting the unnecessary term1 `cosh`/`-1`/`S` path. This is a
score-only change;
the original two-term expression must remain available when a formal certificate is
required. The next gate is matched-budget KL/NLL/PPL, since local and post-`W_o` error
still do not guarantee downstream improvement.

### Stage 3: held-out matched-budget PPL

Setup: Llama-3.1-8B, bfloat16 eager attention, aligned WikiText next-token PPL, block
size 16, first two layers dense, and five non-overlapping 1024-token windows starting
at 5120/6144/7168/8192/9216. These windows were not used in the Stage-2 coefficient
screen. Each method was swept over ten thresholds; NLL was interpolated per window at
the same measured causal attention budget before corpus aggregation. Full-attention
teacher corpus PPL was `6.95138`.

| budget | original PPL | mass-exp, w=1 | term2 w=0.25 | term2 w=0.50 | term1 only | term2 only |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 14.8628 | 1.0001x | 1.1881x | 0.9917x | 2.3217x | 11.3672x |
| 0.25 | 8.2616 | 1.0001x | 1.0600x | 0.9700x | 2.0677x | 13.6354x |
| 0.30 | 7.1929 | 0.9999x | 1.0114x | 1.0178x | 1.6404x | 11.2646x |
| 0.35 | 7.0911 | 1.0002x | 0.9967x | 0.9970x | 1.3512x | 8.7229x |

Entries after the original PPL are corpus-PPL ratios against the original score at
the same budget. The mass-exp simplification is indistinguishable from the original:
its ratio stays within `0.99988-1.00015`, and every paired delta-NLL confidence
interval contains zero with endpoints below `0.0013` in magnitude. This validates
removing term1's explicit `cosh`, `-1`, and `S` path while retaining term2 weight 1.

The Stage-2 `0.922x` post-`W_o` result for term2 weight 0.25 does not transfer to PPL.
Weight 0.25 is 6.0% worse at budget 0.25 and 18.8% worse at budget 0.20. Weight 0.50
is 3.0% better at budget 0.25, but the paired delta-NLL 95% interval is
`[-0.07065, 0.00978]`; it is slightly worse at budget 0.30. No downweighted rule has
a stable or statistically resolved PPL advantage on these five windows.

Both terms are necessary for the end-to-end selector. Term1-only is 1.35-2.32x worse
and term2-only is 8.72-13.64x worse over budgets 0.20-0.35; their paired delta-NLL
intervals are strictly positive. Term1 is the stronger local ranking channel, but
term2 still changes selections that matter after repeated layer composition. This is
the concrete downstream gap that the local and post-`W_o` proxies did not capture.

**Stage-3 decision:** deploy the certified mass-exp replacement with term2 weight 1.
Do not promote the 0.25 coefficient from the local proxy result. Treat 0.50 as an
unconfirmed tuning lead, not a default; a larger held-out run would be needed before
claiming a PPL win. Raw summaries and the full paired table are under
`llama/result/term_ablation_ppl/Llama-3.1-8B/aligned_heldout_n5_s1024_b16/`.
