# Condition Block Speed Notes

This note summarizes the speed work for generation-time `condition_block`.
The current fast implementation is:

```text
llama/script/analysis/generate/methods/condition_block.py
```

The previous full debug / profiling implementation was backed up to:

```text
llama/script/analysis/generate/methods/condition_block_backup.py
```

## Goal

The memory bottleneck was already fixed by changing generation semantics:

```text
prefill: full attention
decode: only the current generated query uses condition-block
```

After that fix, the main problem became speed. GPU utilization and power were
low because the decode implementation launched many tiny operations:

```text
decode step
  x layer
  x KV group
  x condition / token logits / softmax / value reduce kernels
```

For Llama-3.1-8B, each layer has 8 KV heads, so the old fast path still did 8
small condition-block computations per layer per token.

## Optimizations Applied

### 1. Cached Decode

The earliest generate version reran a full forward over:

```text
prompt + generated tokens
```

for every new token. This was correct but extremely slow for long prompts.

The current version uses:

```text
prefill once with use_cache=True
decode one token at a time with past_key_values
```

This changed decode from repeated long-context prefill to normal cached decode.

### 2. Full Prefill Uses SDPA

The prefill stage is full attention, but it is routed through PyTorch SDPA:

```python
F.scaled_dot_product_attention(..., is_causal=True)
```

The important detail is that full-sequence causal attention passes:

```text
attn_mask=None
is_causal=True
```

This is flash-friendly. Passing the HF additive causal mask can force a slower
or more memory-hungry backend.

### 3. Prompt Block Summaries Are Cached

During decode, the prompt is fixed. For each layer, the prompt K/V block
summaries are built once and reused:

```text
k_block
v_block
k_bar
v_bar
k_min
k_max
v_norm_max
block_valid_counts
```

Decode queries are always after the prompt, so every prompt block is fully
visible. The fast path does not need prefix cumsum / cummax / gather for partial
prompt visibility.

### 4. Decode Skips Full SDPA For Compressed Layers

For compressed layers during decode, `q_len=1` and the condition-block output
replaces the full attention output completely. Therefore the code does not first
run full SDPA and then overwrite it.

Full SDPA is still used for:

```text
prefill
non-compressed layers
```

### 5. All KV Groups Are Batched Together

This was the main GPU-utilization fix.

Instead of looping over KV heads:

```text
for kv_head in kv_heads:
    run condition-block for this KV group
```

the current code groups all KV heads into one batched computation:

```text
q_grouped: [num_kv_heads, q_heads_per_kv, 1, head_dim]
k_block:   [num_kv_heads, num_blocks, block_size, head_dim]
```

The token logits are computed with:

```python
torch.einsum("gsqd,gbtd->gsqbt", q_grouped, k_block)
```

This reduces the number of tiny decode kernels by roughly the number of KV
heads. For Llama-3.1-8B, that is an 8x reduction in the condition-block call
count per layer.

### 6. GQA Selection Semantics Are Preserved

For each KV head, multiple query heads first compute their own condition scores.
Then the condition is averaged across the query heads in the same GQA group:

```python
selected = condition.mean(dim=1, keepdim=True) > eps
```

The resulting selected-block mask is shared by all query heads attached to that
KV head.

This matches the intended GQA behavior.

## Current Fast Path Semantics

The simplified `condition_block.py` now keeps only the fastest path:

```text
1. Prefill prompt with full SDPA attention and KV cache.
2. Generate the first token from the prefill logits.
3. For each later token:
   - input only the previous generated token
   - use past_key_values
   - for compressed layers, run all-KV batched condition-block
   - for full layers, run SDPA attention
4. Record equivalent budget metadata.
```

The debug/profiling/fallback paths were removed from the main file and preserved
in `condition_block_backup.py`.

## Speed Results

Representative measurements on Llama-3.1-8B-Instruct, LongBench NarrativeQA,
`block_size=16`, `eps=0.1`.

### Early Memory-Safe Version

```text
NarrativeQA first sample, max_new_tokens=1:
~89.8s/sample
```

Cause: every token effectively involved a long full forward plus condition work.

### Cached Decode + Single-Query Fast Path

```text
NarrativeQA first 10 samples:
~6.04s/sample
```

This introduced cached decode, prompt-summary caching, and the single-query
vectorized condition-block path.

### Cached Decode + Prefix Cache + Skip Compressed-Layer SDPA

```text
NarrativeQA first 10 samples:
~4.92s/sample
```

This avoided unnecessary compressed-layer full SDPA in decode and reused prompt
block summaries.

### Current All-KV Batched Fast Path

```text
NarrativeQA first 10 samples:
~2.83s/sample
wall time: ~41.10s
```

For a smaller controlled probe:

```text
NarrativeQA limit=3, max_new_tokens=16
previous prefix-cache path: ~5.43s/sample
current batched path:       ~2.92s/sample
```

## Profiling Takeaway

Before all-KV batching, one profile showed the decode path dominated by many
small repeated operations:

```text
condition_selection
hybrid_token_logits
hybrid_softmax
hybrid_value_reduce
hybrid_suffix_logits
hybrid_suffix_reduce
```

The count was:

```text
decode_steps * layers * kv_heads
```

After batching all KV heads together, the count became:

```text
decode_steps * layers
```

This was the main reason GPU utilization improved.

## Run Commands

Run one LongBench dataset:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m llama.script.analysis.generate.longbench.run_all \
  --datasets narrativeqa \
  --method condition_block \
  --condition-block-size 16 \
  --condition-eps 0.1
```

Sweep eps values:

```bash
llama/script/analysis/generate/longbench/run_condition_block_eps.sh 0.05 0.1 0.25 0.5
```

Restrict the sweep to one dataset:

```bash
EXTRA_ARGS="--datasets narrativeqa" \
llama/script/analysis/generate/longbench/run_condition_block_eps.sh 0.05 0.1 0.25 0.5
```

Small smoke test:

```bash
EXTRA_ARGS="--datasets narrativeqa --limit 1 --max-new-tokens 4" \
llama/script/analysis/generate/longbench/run_condition_block_eps.sh 0.1
```

## Notes On Numerical Matching

The all-KV batched path preserves the same mathematical GQA condition rule, but
batched reductions can change floating-point order slightly. In greedy decoding,
small logit differences may occasionally lead to different tokens.

In earlier checks:

```text
tiny CPU sanity: matched
NarrativeQA 3-sample probe: predictions matched
NarrativeQA 10-sample comparison: 9/10 exact-match vs previous path
```

So the fast path should be treated as the current optimized implementation, not
as a bitwise-identical refactor of the older fallback implementation.

## Possible Next Optimizations

The remaining work is mostly kernel fusion:

```text
condition score
selection
token/cluster softmax
value reduction
```

Today these are still expressed as PyTorch tensor ops. A Triton kernel or custom
CUDA kernel could fuse the decode condition-block computation and reduce memory
traffic further.

FlexAttention may be useful if the condition-block mask can be expressed as a
block-sparse attention pattern known before attention. The current algorithm
computes selection from Q/K/V summaries at decode time, then mixes selected
token blocks with unselected cluster tokens, so it is not a direct drop-in
replacement for standard block-sparse attention.
