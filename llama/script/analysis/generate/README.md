# Generation Benchmarks

This folder contains generation-style runners for long-context benchmarks.
Unlike `analysis/multisample`, these scripts call autoregressive generation and
can use KVPress directly during prefill.

## Methods

`--method` supports:

- `full`
- `kvpress_snapkv`
- `kvpress_adakv_snapkv`
- `kvpress_streamllm`
- `attention_topk`
- `h2o`
- `condition_block`
- `quest`

KVPress methods use the installed `kvpress` package. Local methods share a
common generation interface and reuse the existing analysis implementations.
They currently decode with full-forward patching, which is slower but keeps the
semantics aligned with the PPL runners.

`h2o` is backed by KVPress `ObservedAttentionPress`, which requires eager
attention; the runner sets `attn_implementation="eager"` automatically when
`--method h2o` is used unless you explicitly override it.

For `condition_block`, `--budget` is not used. Set `--condition-block-size`
and `--condition-eps`; each output JSONL row records the measured
`condition_block_equiv_budget`.

## LongBench

LongBench v1 is loaded directly from the Hugging Face dataset repo
`THUDM/LongBench`. The runner uses the official prompt templates and per-dataset
maximum generation lengths from the LongBench v1 config.

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.generate.longbench.run \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --dataset hotpotqa \
  --method kvpress_snapkv \
  --budget 0.5
```

Use `--longbench-e` to load the LongBench-E variant for datasets that provide it.
Passing `--data /path/to/file.jsonl` still works as a local override.

To run the default 16-task set (14 English tasks + `lcc` and `repobench-p`) with
one compression method, use:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.generate.longbench.run_all \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --method condition_block \
  --condition-block-size 8 \
  --condition-eps 1.0
```

The wrapper loads the model once, then shows an outer `tqdm` progress bar across
datasets while reusing the per-sample progress bars and resume logic from the
single-dataset runner. Override `--datasets` to run a smaller subset.

## RULER

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.generate.ruler.run \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --data /path/to/ruler.jsonl \
  --method kvpress_adakv_snapkv \
  --budget 0.5 \
  --max-new-tokens 64
```

Inputs can be JSONL or JSON. By default the runner reads `context`, `question`,
`answers`, and `id`; override these with `--context-field`, `--question-field`,
`--answer-field`, and `--id-field`. If your file already has a complete prompt,
pass `--prompt-field`.
