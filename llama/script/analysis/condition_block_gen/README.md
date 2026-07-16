# Condition-block generation

This package contains the autoregressive generation phase for condition-block,
plus the baselines used for generation comparisons. Unlike
`analysis.condition_block_ppl.multisample`, these scripts generate new tokens
instead of computing teacher-forced PPL, and can use KVPress directly during
prefill.

Condition-block is the main method in this package. Attention top-k, H2O,
SnapKV/AdaKV, StreamLLM, QUEST, Double-P, K-similarity and full attention are retained as
generation baselines. PPL runners and bound analysis live in the sibling
`condition_block_ppl/` package.

## Methods

`--method` supports:

- `full`
- `kvpress_snapkv`
- `kvpress_adakv_snapkv`
- `kvpress_streamllm`
- `attention_topk`
- `h2o`
- `condition_block`
- `condition_block_triton`
- `double_p`
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

`condition_block_triton` keeps the same selection semantics but uses a
decode-only Tensor-Core kernel. Unexpanded blocks contribute one representative;
only selected 16-token blocks load token-level K/V. Block size 16 is the tuned
path. Set `CONDITION_BLOCK_SKIP_STATS=1` for timing runs to avoid metadata syncs.

For `quest`, pass exactly one of `--budget` or `--quest-page-size`. When
`--budget` is used, the runner sets `quest_page_size=round(1 / budget)` and
names the output file with the budget.

For `double_p`, use `--double-p-p1`, `--double-p-p2`,
`--double-p-cluster-size`, `--double-p-sink-tokens`, and
`--double-p-window-size`. The default thresholds `0.95/0.70` are the paper's
reported Llama-3.1 setting. This is a cached eager PyTorch accuracy reference:
it uses dense prefill, clusters the prompt middle per KV head, keeps generated
tokens exact, and reports the measured equivalent budget. It does not provide
a Double-P latency/speedup measurement because the paper's custom kernel is not
publicly available here.

Local full-forward methods (`attention_topk`, `condition_block`, and `quest`)
require eager attention so the runner can capture Q/K/V; this is set
automatically unless you explicitly pass another attention implementation.

## Prefill vs Decode Profiling

To test whether a 16K-token prefill plus 512 generated tokens can spend most of
its time in generation, run the cached decode profiler:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.condition_block_gen.profile_prefill_decode \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --prompt-tokens 16384 \
  --decode-steps 512 \
  --repeats 3 \
  --warmup 1
```

The script times one full-context prefill forward separately from 512 cached
single-token decode forwards, then prints `decode_pct`. It appends repeat and
aggregate rows to `llama/result/generate/prefill_decode_profile.jsonl`.

You can sweep context and output lengths in one run:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.condition_block_gen.profile_prefill_decode \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --prompt-tokens 4096 8192 16384 \
  --decode-steps 128 512 \
  --repeats 3
```

## LongBench

LongBench v1 is loaded directly from the Hugging Face dataset repo
`THUDM/LongBench`. The runner uses the official prompt templates and per-dataset
maximum generation lengths from the LongBench v1 config.

The repository's formal setting uses `meta-llama/Llama-3.1-8B-Instruct` in
BF16, seed 42, greedy decoding (`num_beams=1`, no sampling), the complete test
split, and the model's 131,072-token position limit. Prompts longer than the
limit are truncated in the middle before tokenization. The model chat template
is applied except on `trec`, `triviaqa`, `samsum`, `lcc`, and `repobench-p`,
matching the LongBench recommendation for those tasks. Dataset-specific output
limits range from 32 to 512 tokens and come from `dataset2maxlen.json`.

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.condition_block_gen.longbench.run \
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
  python -m script.analysis.condition_block_gen.longbench.run_all \
  --device cuda:0 \
  --model meta-llama/Llama-3.1-8B \
  --method condition_block \
  --condition-block-size 8 \
  --condition-eps 1.0
```

The wrapper loads the model once, then shows an outer `tqdm` progress bar across
datasets while reusing the per-sample progress bars and resume logic from the
single-dataset runner. Override `--datasets` to run a smaller subset.

The reproducible Double-P accuracy run uses the paper's Llama-3.1 thresholds
and the same complete 16-task protocol:

```bash
GPU_ID=0 bash llama/script/analysis/condition_block_gen/longbench/run_double_p.sh
```

Its defaults are `p1=0.95`, `p2=0.70`, cluster size 32, four k-means
iterations, no forced dense early layers, four sink tokens, and a 64-token exact prompt window. Generation
uses dense prompt prefill followed by cached Double-P decode, which is the
native LongBench generation setting (unlike full-chunk teacher-forced PPL).
Each JSONL row records both the combined equivalent budget and enough counts
for `eval_dataset_plot.py` to report a decode-only budget with the dense first
generation step removed.

After all datasets finish, produce the per-task scores, macro average, budget
summary, full-attention comparison, and macro curves against the complete
StreamLLM and SnapKV sweeps with:

```bash
conda run -n nanogpt python -m \
  llama.script.analysis.condition_block_gen.longbench.summarize_double_p
```

The summary reports both the repository-aligned 16-task macro average and the
13-task subset used in the Double-P paper's LongBench table. The repository
protocol adds `trec`, `samsum`, and `passage_count` to that paper subset.

After the per-dataset `eval_dataset_plot.py` summaries exist, generate a clean
4-by-4 task overview plus one presentation plot per task with:

```bash
conda run -n nanogpt python -m \
  llama.script.analysis.condition_block_gen.longbench.plot_double_p_tasks
```

These plots use a log-scale decode budget, show the measured Double-P point and
the dense score, retain the available fixed-budget baselines, and collapse the
available eager/Triton ConditionBlock sweeps into one non-dominated Pareto
frontier. The raw per-dataset plots remain unchanged under each dataset's
`eval_plots/` directory.

## RULER

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.condition_block_gen.ruler.run \
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
