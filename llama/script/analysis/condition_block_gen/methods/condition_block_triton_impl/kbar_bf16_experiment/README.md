# BF16 k_bar storage experiment

This experiment isolates one additional HBM-IO reduction on top of the exact
mixed-summary layout:

```bash
CONDITION_BLOCK_MIXED_SUMMARIES=1 \
CONDITION_BLOCK_K_BAR_DTYPE=bfloat16
```

Only `k_bar` storage changes.  `k_max/k_min`, `v_bar`, value norms, the
condition formula, selected-page attention and suffix attention are unchanged.
The Triton selection kernel converts the quantized mean to FP32 after loading
it, so this is an approximate-storage experiment rather than a new condition
formula.

At 128K with 8 KV heads and head_dim 128, this removes 8 MiB per layer for
block 32 (256 MiB over 32 layers), or 4 MiB per layer for block 64 (128 MiB
over 32 layers).  The same bytes are avoided in every decode-step selection
stream.

Run the synthetic paired sanity check with:

```bash
PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.kbar_bf16_experiment.sanity \
  --device cuda:0
```

Acceptance gates:

1. Quantify `s`, `delta`, selected-mask and fused-output changes for block 32
   and 64, both ordinary and TMA bounds paths.
2. Run paired NarrativeQA and GovReport quality against FP32 `k_bar`.
3. Measure cold-L2 selection/fused latency at 64K/128K.
4. Measure CUDA-graph decode-only latency with stats disabled.  Keep the path
   opt-in unless quality is stable and end-to-end gain exceeds run noise.

## Results

Synthetic paired sanity (block 32/64, ordinary/TMA bounds):

- selected mask mismatch: `0` in all four cases;
- `s` / `delta` max-abs: `0.0016–0.0027`;
- fused BF16 output max-abs: at most `2.44e-4`.

LongBench quality, block 64, eps 0.1, 20 samples per dataset:

| dataset | FP32 score | BF16 score | delta | exact predictions | budget FP32→BF16 |
|---|---:|---:|---:|---:|---:|
| NarrativeQA | 38.84 | 38.84 | 0.00 | 18/20 | 0.2460→0.2454 |
| GovReport | 24.22 | 24.27 | +0.05 | 6/20 | 0.28453→0.28451 |

Cold-L2 stage latency, exact mixed baseline to BF16 `k_bar`:

| block/context | selection | speedup | production fused | speedup |
|---|---:|---:|---:|---:|
| 32 / 64K | 39.10→33.11 us | 1.181x | 60.66→55.61 us | 1.091x |
| 32 / 128K | 55.43→49.94 us | 1.110x | 82.09→77.57 us | 1.058x |
| 64 / 64K | 30.72→28.41 us | 1.081x | 49.16→46.59 us | 1.055x |
| 64 / 128K | 39.36→33.30 us | 1.182x | 61.28→56.06 us | 1.093x |

TMA does not compose profitably with this layout: block64 production fused is
1.001x at 64K and 0.991x at 128K.

CUDA-graph decode-only, block64, 128 fixed tokens, one warmup plus three
measured samples:

| context | FP32 k_bar | BF16 k_bar | speedup |
|---:|---:|---:|---:|
| 64K | 2.0589 s | 2.0452 s | 1.007x |
| 128K | 2.1072 s | 2.0928 s | 1.007x |

Decision: keep the implementation opt-in.  It has a clear attention-stage and
memory-footprint benefit, but its end-to-end latency change is close to noise
and it is not bitwise equivalent.  Do not enable it by default or stack it with
TMA bounds based on these measurements.
