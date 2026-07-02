# FlexAttention condition-block decode experiments

This folder contains isolated FlexAttention experiments for condition-block
decode attention. These scripts do not change the production
`condition_block_triton` path.

## Goal

Evaluate whether PyTorch FlexAttention's inference/decoding path can replace or
accelerate the current custom Triton condition-block decode attention.

The method we want to approximate is:

```text
unselected prompt cluster -> one representative Kbar/Vbar
selected prompt cluster   -> original token-level K/V
generated suffix          -> exact token-level K/V
```

## Scripts

- `condition_block_flex_stage_latency.py`
  Synthetic stage benchmark for full SDPA, Flex dense decode,
  Flex BlockMask token sparsity, Flex compact hybrid attention, and the current
  Triton dummy hybrid stage.
- `summarize_condition_block_flex_stage_latency.py`
  Prints compact tables from the JSONL benchmark output.

## Important FlexAttention findings

FlexAttention supports inference/decode through FlexDecoding, but compile mode
matters:

```text
torch.compile(..., mode="reduce-overhead"):
  flex_block_size=32 -> Inductor NoValidChoicesError

torch.compile(..., dynamic=True, mode="max-autotune"):
  flex_block_size=32  -> works
  flex_block_size=128 -> works
```

The benchmark resets Dynamo and compiles per context length. Without this, a
32K FlexDecoding graph can leak KV-length assumptions into later 64K/128K runs.

`BlockMask.from_kv_blocks` is much closer to decode-time routing than
`create_block_mask(mask_mod)`, because it directly builds sparse KV block
metadata instead of scanning the full mask function.

## Commands

Block size 32 experiment:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.generate.flexattention_experiment.condition_block_flex_stage_latency \
  --device cuda:0 \
  --contexts 32768 65536 131072 \
  --condition-block-size 32 \
  --flex-block-size 32 \
  --suffix-tokens 128 \
  --selected-ratios 0 0.1 0.25 \
  --compile-mode max-autotune \
  --warmup 10 \
  --iters 50 \
  --output /tmp/condition_block_flex_stage_latency_b32_flex32.jsonl
```

Block size 128 experiment:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.generate.flexattention_experiment.condition_block_flex_stage_latency \
  --device cuda:0 \
  --contexts 32768 65536 131072 \
  --condition-block-size 128 \
  --flex-block-size 128 \
  --suffix-tokens 128 \
  --selected-ratios 0 0.1 0.25 \
  --compile-mode max-autotune \
  --warmup 10 \
  --iters 50 \
  --output /tmp/condition_block_flex_stage_latency_b128_flex128.jsonl
```

Summarize:

```bash
python llama/script/analysis/generate/flexattention_experiment/summarize_condition_block_flex_stage_latency.py \
  /tmp/condition_block_flex_stage_latency_b32_flex32.jsonl
```

## Sanity

BF16, Llama-8B attention shape:

```text
B32 compact Flex vs dense reference max error:      4.8e-4
B32 compact Flex vs Triton dummy hybrid max error:  9.8e-4

B128 compact Flex vs dense reference max error:     5.9e-4
B128 compact Flex vs Triton dummy hybrid max error: 9.8e-4
```

## Results

Stage latency, `suffix_tokens=128`, BF16, Llama-8B attention shape.

### `condition_block_size=32`, `flex_block_size=32`

```text
context  selected  SDPA   Flex dense  Triton hybrid  Flex compact  Flex BlockMask
32K      0%        0.269  0.271       0.029          0.057         0.217 ms
32K      10%       0.269  0.271       0.029          0.055         0.234 ms
32K      25%       0.269  0.271       0.029          0.054         0.225 ms

64K      0%        0.574  0.568       0.028          0.056         0.401 ms
64K      10%       0.574  0.568       0.031          0.059         0.435 ms
64K      25%       0.574  0.568       0.040          0.106         0.435 ms

128K     0%        1.137  1.134       0.032          0.056         0.768 ms
128K     10%       1.137  1.134       0.043          0.083         0.840 ms
128K     25%       1.137  1.134       0.123          0.269         0.849 ms
```

For 10% selected, `BlockMask.from_kv_blocks` avoids most Python mask-build
cost:

```text
context  create_block_mask build  from_kv_blocks build  from_kv latency
32K      2.28 ms                  0.053 ms              0.218 ms
64K      2.42 ms                  0.053 ms              0.416 ms
128K     2.24 ms                  0.054 ms              0.814 ms
```

### `condition_block_size=128`, `flex_block_size=128`

```text
context  selected  SDPA   Flex dense  Triton hybrid  Flex compact  Flex BlockMask
32K      0%        0.267  0.271       0.027          0.055         0.218 ms
32K      10%       0.267  0.271       0.031          0.055         0.243 ms
32K      25%       0.267  0.271       0.043          0.054         0.242 ms

64K      0%        0.571  0.568       0.028          0.056         0.406 ms
64K      10%       0.571  0.568       0.033          0.057         0.465 ms
64K      25%       0.571  0.568       0.046          0.078         0.463 ms

128K     0%        1.136  1.134       0.028          0.054         0.771 ms
128K     10%       1.136  1.134       0.045          0.055         0.902 ms
128K     25%       1.136  1.134       0.109          0.246         0.898 ms
```

Compact K/V build cost is not included in `Flex compact` kernel latency. It is
too high for per-layer, per-token decode:

```text
B32:
32K   31-43 ms
64K   62-75 ms
128K  124-147 ms

B128:
32K   7-9 ms
64K   15-17 ms
128K  29-32 ms
```

## Conclusion

- FlexAttention is usable for inference decode, especially with
  `mode="max-autotune"`.
- Flex dense decode is roughly the same speed as SDPA in this setup.
- Flex BlockMask can skip token blocks, but it has no representative semantics:
  skipped prompt clusters are simply absent, not replaced by Kbar/Vbar.
- Flex compact hybrid can match the condition-block representative semantics,
  but requires dynamic compact K/V packing. That packing dominates latency.
- The current custom Triton hybrid stage remains much faster for this method.

Practical recommendation: keep FlexAttention as a reference / exploration path,
but keep the production acceleration on the custom Triton or paged-attention
style implementation.
