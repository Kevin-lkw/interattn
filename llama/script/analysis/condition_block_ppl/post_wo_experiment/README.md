# Post-Wo condition-block experiment

This directory is self-contained. It imports shared model-loading/capture helpers but
does not modify the existing condition-block implementation.

## Variants

- `pre_wo`: original value norms `max ||v_i||`.
- `post_wo_spectral`: certified `||W_h||_2 max ||v_i||`.
- `post_wo_exact`: certified `max ||W_h v_i||`, computed through
  `v_i^T W_h^T W_h v_i` without materializing hidden-size projected values.

For GQA, each condition is computed per query head and averaged inside its KV group,
matching the existing shared-block selection rule. The average differs from the
theoretical head sum only by the constant GQA group size.

## Commands

Run from `llama/` in the `nanogpt` environment:

```bash
PYTHONPATH=. python -m script.analysis.condition_block_ppl.post_wo_experiment.correctness \
  --trials 2000

PYTHONPATH=. python -m script.analysis.condition_block_ppl.post_wo_experiment.offline_spearman \
  --device cuda:1 --dtype float32 --seq-len 1024 --block-size 10 \
  --layers 10 15 20 --num-queries 16

PYTHONPATH=. python -m script.analysis.condition_block_ppl.post_wo_experiment.run_ppl \
  --device cuda:1 --dtype float32 --seq-len 1024 --num-samples 1 \
  --block-size 10 --eps 0.1 0.25 0.5 1.0 \
  --variants pre_wo post_wo_spectral post_wo_exact \
  --full-attention-layers 2 --ppl-protocol legacy

PYTHONPATH=. python -m script.analysis.condition_block_ppl.post_wo_experiment.merge_ppl_shards

PYTHONPATH=. python -m script.analysis.condition_block_ppl.post_wo_experiment.summarize_results
```

The PPL runner always requires the first two layers to use full attention. Every
subsequent layer uses the selected condition variant and the same contiguous-block
hybrid computation.

## Completed setting

- Model: `meta-llama/Llama-3.1-8B`
- Data: five non-overlapping WikiText windows, starting at tokens
  0/1024/2048/3072/4096, each of length 1024
- Dtype: float32
- Block size: 10
- Delta: coordinate range bound
- Full-attention layers: 0 and 1
- Compressed layers: 2 through 31
- Epsilon sweep: 0.1, 0.25, 0.5, 1.0
- Offline correlation: layers 10/15/20, 16 queries from positions 256 through 1023,
  all 32 query heads

Machine-readable outputs are under `llama/result/post_wo_condition_block/`;
`llama/result/post_wo_condition_block/report.json` is the compact combined report.
The five-window PPL aggregate and plot are under
`llama/result/post_wo_condition_block/ppl_n5/`; the individual resumable runs are
retained under `ppl/` and `ppl_shards/`.
The analysis source directory never contains experiment results.
