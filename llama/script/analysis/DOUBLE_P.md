# Double-P implementation map

This repository contains one shared eager-PyTorch Double-P attention core and
three evaluation protocols.  None of these paths is a Triton/fused latency
implementation; timing from them must not be reported as Double-P speedup.

## Algorithm and shared invariants

The shared implementation is
`condition_block_gen/methods/double_p.py`:

1. Exclude the exact sink prefix and local prompt window from clustering.
2. Cluster the remaining prompt keys independently per layer and KV head with
   deterministic FP32 Euclidean k-means.
3. Score each cluster with `q @ mean_k / sqrt(d) + log(cluster_count)`.
4. Keep the smallest descending cluster prefix whose probability mass reaches
   `p1`.
5. Expand the nested `p2` prefix to exact tokens, represent the rest of the
   `p1` prefix by `(mean_k, mean_v)`, and prune clusters outside `p1`.
6. Normalize exact tokens and approximate cluster representatives together in
   one softmax denominator.

Thresholds must satisfy `0 < p2 <= p1 <= 1`.  `(1,1)` is the dense-equivalent
correctness anchor.  The paper-selected Llama-3.1 setting is `(0.95,0.70)`.

The stable public helpers are:

- `build_double_p_prompt_clusters`: prompt partitioning and cluster summaries.
- `double_p_attention`: mixed exact/centroid attention and work accounting.
- `DoublePDecodeRunner` / `generate_double_p_cached`: cached generation path.
- `top_p_mask`: minimal probability-prefix selection.

## Protocols

| Protocol | Entry point | Dense prefill | Scored queries | Clustering behavior |
| --- | --- | --- | --- | --- |
| LongBench generation | `condition_block_gen/methods/double_p.py` | Yes | Autoregressive decode | Cluster the completed prompt once per layer/KV head and cache it |
| Decode-faithful PPL | `condition_block_ppl/runner_double_p.py` | Yes | Teacher-forced continuation only | Cluster the fixed prompt; continuation is never visible to prompt k-means |
| Full-causal PPL extension | `condition_block_ppl/runner_double_p_full_causal.py` | No fixed prefill | Every fixed-chunk causal query | Rebuild clusters after each complete `cluster_size` group leaves the exact window |

LongBench and decode-faithful PPL match the paper's prompt-then-decode use
case.  Full-causal PPL is a repository extension for alignment with
traditional fixed-chunk perplexity; it should be named explicitly in tables.

In cached generation, sink tokens, the prompt-tail window, and all generated
tokens remain exact.  The first generated token comes from dense prefill
logits.  Decode-only budget reports remove that unavoidable first step.

## Configuration source of truth

`condition_block_ppl/double_p_config.py` owns:

- threshold parsing and validation;
- stable PPL result keys;
- the paper and dense-anchor settings;
- decode-faithful and full-causal default threshold grids;
- validation that PPL grids have unique `p1` values.

The unique-`p1` constraint is an output-schema requirement: multisample PPL
summaries use `p1` as their setting key and store `p2` alongside it.

## LongBench workflow

From the repository root:

```bash
GPU_ID=0 bash \
  llama/script/analysis/condition_block_gen/longbench/run_double_p.sh

conda run -n nanogpt python -m \
  llama.script.analysis.condition_block_gen.longbench.summarize_double_p_sweep

conda run -n nanogpt python -m \
  llama.script.analysis.condition_block_gen.longbench.plot_double_p_tasks
```

Related files:

- `longbench/summarize_double_p.py`: one-setting report.
- `longbench/summarize_double_p_sweep.py`: paper-threshold sweep, CSVs, and
  macro plots.
- `longbench/plot_double_p_tasks.py`: one comparison plot per task and the
  joint ConditionBlock/Double-P Pareto frontier.

## PPL workflows

Decode-faithful single-sample diagnostic:

```bash
python -m llama.script.analysis.condition_block_ppl.runner_double_p \
  --model meta-llama/Llama-3.1-8B \
  --seq-len 1024 --prompt-len 768 \
  --p-settings 0.90:0.50 0.95:0.70 1.0:1.0
```

Traditional full-chunk aligned PPL uses the full-causal extension:

```bash
python -m \
  llama.script.analysis.condition_block_ppl.multisample.run_double_p_full_causal \
  --help
```

The decode-faithful multisample counterpart is
`condition_block_ppl/multisample/run_double_p.py`.

## Verification

```bash
python -m pytest -q \
  llama/script/analysis/condition_block_ppl/test_double_p.py

python -m py_compile \
  llama/script/analysis/condition_block_gen/methods/double_p.py \
  llama/script/analysis/condition_block_ppl/double_p_config.py \
  llama/script/analysis/condition_block_ppl/runner_double_p.py \
  llama/script/analysis/condition_block_ppl/runner_double_p_full_causal.py
```

The unit tests cover top-p prefix semantics, k-means summary conservation,
dense-equivalent GQA attention, full-chunk budget composition, causal cluster
boundaries, full-causal dense equivalence, and threshold-grid validation.
