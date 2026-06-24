# Condition Block Speed Summary

This file summarizes the main speed optimizations for generation-time
`condition_block`.

## Main Problem

After the memory issue was fixed, inference was still slow because decode used
many small GPU operations. GPU utilization and power were low.

The slow pattern was roughly:

```text
decode step x layer x KV group
```

For Llama-3.1-8B, each layer has 8 KV heads, so the old version launched many
small condition-block computations per token.

## What Changed

The current fast implementation is:

```text
llama/script/analysis/generate/methods/condition_block.py
```

The previous full/debug version is backed up at:

```text
llama/script/analysis/generate/methods/condition_block_backup.py
```

Key optimizations:

1. **Use KV cache**

   Prefill runs once. Decode only processes one new token at a time.

2. **Prefill uses SDPA / flash-friendly full attention**

   Prompt tokens use normal full attention, but through PyTorch SDPA.

3. **Cache prompt block summaries**

   Prompt K/V block summaries are fixed during decode, so each layer builds
   them once and reuses them.

4. **Skip unnecessary full attention in compressed decode layers**

   For compressed layers, the decode query output is fully replaced by
   condition-block attention, so we do not compute full attention first.

5. **Batch all KV heads together**

   Instead of running condition-block separately for each KV head, the current
   version computes all KV groups in one batched operation.

This last change is the main GPU-utilization improvement.

## Current Semantics

```text
prefill:
  full attention
  build KV cache

first token:
  generated from full prefill logits

later decode tokens:
  q_len = 1
  compressed layers use condition-block
  full layers use SDPA full attention
```

The GQA rule is preserved: different Q heads under the same KV head compute
condition scores, average them, and share the selected blocks.

## Speed Results

Representative numbers on Llama-3.1-8B-Instruct, NarrativeQA,
`block_size=16`, `eps=0.1`:

```text
early memory-safe version:         ~89.8s/sample for first sample, 1 token
cached + single-query fast path:   ~6.04s/sample on first 10 samples
prefix cache + skip full SDPA:     ~4.92s/sample on first 10 samples
current all-KV batched path:       ~2.83s/sample on first 10 samples
```

Controlled short probe:

```text
NarrativeQA limit=3, max_new_tokens=16
previous path: ~5.43s/sample
current path:  ~2.92s/sample
```

## How To Run

Single dataset:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m llama.script.analysis.generate.longbench.run_all \
  --datasets narrativeqa \
  --method condition_block \
  --condition-block-size 16 \
  --condition-eps 0.1
```

Sweep eps:

```bash
llama/script/analysis/generate/longbench/run_condition_block_eps.sh 0.05 0.1 0.25 0.5
```

Smoke test:

```bash
EXTRA_ARGS="--datasets narrativeqa --limit 1 --max-new-tokens 4" \
llama/script/analysis/generate/longbench/run_condition_block_eps.sh 0.1
```

## Note

The current fast path is optimized for speed and GPU utilization. It preserves
the intended algorithm, but because operations are batched differently, greedy
decoding may occasionally differ from older implementations due to small
floating-point differences.

Future speed work should focus on fusing the condition-block decode computation
with Triton or a custom CUDA kernel.
