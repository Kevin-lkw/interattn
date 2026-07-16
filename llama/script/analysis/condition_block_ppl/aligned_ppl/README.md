# Aligned WikiText-2 PPL

This suite implements the fixed-chunk perplexity protocol commonly used in
LLaMA compression and quantization work. It intentionally does not implement
the decode-tail experiment.

## Protocol

- Model: `meta-llama/Llama-3.1-8B`.
- Dataset: `wikitext-2-raw-v1`, test split.
- Text construction: preserve the raw `text` column, including empty rows, and
  join every row with `\n\n` (the convention used by common WikiText-2
  compression loaders).
- Tokenization: the model tokenizer with special tokens enabled once for the
  concatenated stream.
- Evaluation chunks: the default experiment uses the first 20 non-overlapping
  chunks of 2048 tokens, starting at `0, 2048, ..., 38912`. This is a
  deterministic 40,960-token subset of the test stream, not full-test PPL. It
  contributes 40,940 scored next-token targets. The full token stream can form
  141 complete chunks if `NUM_SAMPLES=141` is requested. Under this raw-column
  construction the complete test stream has 289,077 Llama-3.1 tokens and a
  309-token remainder after the 141 complete chunks.
- Cache: `use_cache=False`; every chunk is an independent full causal forward.
- Loss: standard within-chunk shift, `logits[:, :-1]` against
  `input_ids[:, 1:]`. Thus every complete chunk contributes 2047 targets, and
  no label is read from the following chunk.
- Aggregate: token-weighted mean NLL followed by one exponentiation
  (`corpus_ppl`). The arithmetic mean of per-chunk PPL is retained only as a
  diagnostic and is not plotted as the reported PPL.
- Precision: bfloat16 model forward; logits are converted to float32 before
  NLL, KL, and routing diagnostics.
- Plot: measured equivalent causal attention budget on a logarithmic x-axis;
  corpus PPL on a logarithmic y-axis.

The default method grids are inherited from `multisample/`:

- Oracle attention top-k: budgets `0.01 0.05 0.1 0.25 0.5 1.0`; first two
  layers dense.
- Condition-block: block size 10, range-bound delta, eps
  `1.0 0.5 0.25 0.1 0.075 0.05`; first two layers dense.
- StreamLLM: budgets `0.25 0.5 1.0`; four sink tokens through the shared sink
  implementation, every layer compressed.

The current n=20 run intentionally omits H2O and QUEST. Their shared runners
remain available, but they are not required by `aligned_ppl/run_all.sh` and are
not included in the combined plot.

Double-P is excluded because its published algorithm is defined for sparse
decode after dense prefill; applying it to every prefill query would be an
unpublished extension rather than the traditional fixed-chunk protocol.

## Sanity check

Run before the experiment:

```bash
cd llama
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.condition_block_ppl.aligned_ppl.sanity_check \
  --device cuda:0
```

The check verifies chunk boundaries, within-window labels, target count,
agreement with the Hugging Face causal-LM loss, and exact dense self-KL/NLL.
The standalone command uses a short 256-token check; `run_all.sh` passes the
formal 2048-token sequence length. JSON reports are stored under
`result/Llama-3.1-8B/aligned_ppl/sanity/`.

## Full run

```bash
cd llama
GPU_ID=7 bash script/analysis/condition_block_ppl/aligned_ppl/run_all.sh
```

The runners save after every chunk and resume completed chunks. Override any
parameter through environment variables; `NUM_SAMPLES=141` runs all complete
chunks. The default n=20 subset outputs are written to:

```text
result/Llama-3.1-8B/aligned_ppl/
  README.md
  sanity/sanity.json
  logs/*.log
  <method>/summary.pt
  <method>/corpus_ppl_vs_budget.png
  all_methods_corpus_ppl_vs_budget.png
```

For sample-parallel execution, disjoint outputs can be merged by token start
with `aligned_ppl.merge_summaries`. The merge rejects duplicate or missing
starts, mismatched methods, and mismatched setting grids before recomputing the
token-weighted aggregate.
