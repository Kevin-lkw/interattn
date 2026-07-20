# Real LongBench attention capture benchmark

This benchmark replaces the old random synthetic routing measurement with real
model tensors captured from LongBench-v2. It runs one unchanged SDPA prefill,
captures the first condition-block decode step, and replays three attention
paths on the same Q/K/V and selected mask:

1. contiguous full SDPA decode attention;
2. production fused Triton condition selection + hybrid attention;
3. fixed-real-mask hybrid attention, which skips condition selection but still
   computes representative QK, selected-page attention, exact suffix attention,
   online softmax and value reduction.

The fixed-mask path is an executable upper bound. A separate optimistic IO bound
is calculated from the actual per-layer selected masks. Raw tensors are consumed
in GPU memory and not written to disk.

Example on one exclusive GPU:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/scratch1/liankewei/interattn \
/scratch1/liankewei/miniconda3/envs/nanogpt/bin/python -m \
  llama.script.analysis.condition_block_gen.real_attention_benchmark.run \
  --device cuda:0 --contexts 32768 65536 131072 \
  --block-sizes 64 --eps 0.1 --samples 2
```

Outputs:

- `layers.jsonl`: every real sample/layer capture;
- `summary.json`: aggregated machine-readable results;
- `RESULTS.md`: attention speedup, theoretical bound and realized fractions.

Latency uses CUDA events and cold-L2 replay. It excludes prefill, model GEMV,
cache update and Python dispatch. The production selected mask must match the
separately captured mask exactly, and fixed-mask output must remain within BF16
numerical tolerance of the production fused output.

## Measured result

RTX PRO 6000 Blackwell, GPU 6 exclusive, Llama-3.1-8B-Instruct, block 64,
eps 0.1, mixed summaries plus BF16 `k_bar`, five LongBench-v2 records per
context, all 32 layers, first sparse decode step, 10 warmups + 30 cold-L2
measurements per capture:

| context | captures | selected blocks | full | production | fixed real mask | production speedup | fixed ceiling | optimistic IO ideal | IO ideal realized | runnable ceiling realized |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 160 | 3.21% | 142.13 us | 58.78 us | 43.14 us | 2.42x | 3.29x | 15.84x | 15.3% | 73.4% |
| 64K | 160 | 1.55% | 251.68 us | 68.00 us | 44.96 us | 3.70x | 5.60x | 21.42x | 17.3% | 66.1% |
| 128K | 160 | 0.95% | 438.03 us | 80.41 us | 53.97 us | 5.45x | 8.12x | 24.60x | 22.1% | 67.1% |

The ratios correspond to about 16.4, 15.9 and 19.4 selected blocks per KV
head: the absolute routing budget stays roughly constant as context grows.
Production realizes about two thirds of the executable fixed-mask ceiling.
The remaining production-to-fixed gap is condition selection; the much larger
fixed-to-IO-ideal gap is the cost of small dependent kernels, representative
QK, online softmax/reduction, partial-buffer traffic and the GPU latency floor.

Sanity across all 480 captures:

- separately materialized routing equals fused production routing exactly;
- fixed-mask vs production BF16 output max-abs is at most `9.77e-4`;
- no synthetic random tensors are used for routing or latency inputs.

Raw output:
`llama/result/generate/condition_block_real_attention_benchmark_5samples/`.
