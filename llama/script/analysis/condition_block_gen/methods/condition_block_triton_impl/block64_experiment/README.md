# Native block-64 experiment

This directory records validation for the 64-token page extension of the
production fused Triton path.  The range-bound and hybrid-attention formulas
are unchanged; only the prompt partition changes from 32-token to 64-token
blocks.

The production page kernel already parameterizes `PAGE_SIZE`.  A selected
64-token page is therefore loaded directly and consumed by one logical
`q @ K_page` / `P @ V_page` tile inside the fused kernel.  No Python-level
split into block-16 or block-32 calls is used.

Synthetic dense-reference sanity:

```bash
PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.block64_experiment.sanity \
  --device cuda:0
```

Acceptance gates:

1. Selected masks must match the existing Triton materialization path.
2. Fused outputs must match the dense hybrid-attention reference for fully
   selected, fully representative, padded and non-padded prompts.
3. Real LongBench generation must complete without NaN/OOM and quality must be
   compared against block 32.
4. 64K/128K decode-only or attention-stage latency must improve before block 64
   is recommended as a long-context configuration.

## Results

Synthetic sanity (GPU 7): selected masks match exactly for every case.  BF16
output max-abs is `2.44e-4` to `4.88e-4`; the difference is expected from the
fused online-softmax reduction order.

LongBench quality, `eps=0.1`, 20 samples per dataset:

| dataset | block32 score | block64 score | delta | block32 budget | block64 budget |
|---|---:|---:|---:|---:|---:|
| NarrativeQA | 38.66 | 38.84 | +0.18 | 0.207 | 0.246 |
| GovReport | 24.51 | 24.22 | -0.29 | 0.176 | 0.285 |

The score is stable on this sanity set, but block 64 selects materially more
token mass at the same epsilon.  Quality runs keep stats enabled and are not
used as latency measurements.

Cold-L2 stage latency:

| context | stage | block32 | block64 | speedup |
|---:|---|---:|---:|---:|
| 64K | selection stats+reduce | 40.50 us | 33.97 us | 1.192x |
| 64K | production fused | 65.77 us | 54.99 us | 1.196x |
| 64K | dummy 0% | 44.96 us | 33.74 us | 1.332x |
| 128K | selection stats+reduce | 63.55 us | 41.35 us | 1.537x |
| 128K | production fused | 96.04 us | 66.04 us | 1.454x |
| 128K | dummy 0% | 64.19 us | 45.25 us | 1.418x |

The synthetic production routing is useful only as a kernel regression; its
random summaries do not reproduce the selected ratios of a real model.

CUDA-graph decode-only, LongBench-v2, 128 fixed tokens, stats disabled, one
warmup plus three measured samples:

| context | block32 | block64 | speedup |
|---:|---:|---:|---:|
| 64K | 2.0613 s | 2.0662 s | 0.998x |
| 128K | 2.1850 s | 2.1296 s | 1.026x |

Decision: retain native block 64 as an explicit long-context option, but keep
block 32 as the default.  At 64K there is no end-to-end gain; at 128K the
reduced representative/bound IO survives dilution and gives a 2.6% gain.
