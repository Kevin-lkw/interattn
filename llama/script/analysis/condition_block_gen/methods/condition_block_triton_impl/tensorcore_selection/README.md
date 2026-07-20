# Tensor Core condition-selection experiment

This directory contains an isolated prototype for accelerating the exact
coordinate range-bound formulation.  It does not change the production path.

The algebraic rewrite is:

```text
q_pos = max(q, 0)
q_neg = min(q, 0)

s     = q_pos @ k_bar + q_neg @ k_bar
upper = q_pos @ k_max + q_neg @ k_min
lower = q_pos @ k_min + q_neg @ k_max
delta = max(abs(upper - s), abs(lower - s))
```

The implementation places `[q, q_pos, q_neg, 0]` and
`[k_bar, k_max, k_min, 0]` in one padded `16 x 128 x 64` `tl.dot` tile.  This
does more arithmetic than the scalar sign-select kernel, but attempts to trade
that arithmetic for Tensor Core throughput without changing the mathematical
bound.

Precision modes:

- `ieee`: FP32 matrix multiply without TF32 approximation;
- `tf32x3`: three-term TF32 decomposition, intended as the accuracy/speed
  compromise;
- `tf32`: fastest Tensor Core FP32-input mode, with the largest numerical
  perturbation.

Run the isolated cold-L2 benchmark:

```bash
PYTHONPATH=/scratch1/liankewei/interattn \
python -m llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.tensorcore_selection.benchmark \
  --device cuda:0 --contexts 32768 65536 131072 --cold-l2
```

Acceptance gates before any production integration:

1. `s`, `delta`, global normalizers and selected masks must be compared against
   the existing Triton selection kernel.
2. The new kernel must be faster under cold-L2 at 64K and 128K.
3. A surviving precision mode must pass fused-output and LongBench generation
   sanity after opt-in integration.

## Results

Hardware and setting:

```text
GPU: RTX PRO 6000 Blackwell Server Edition, exclusive GPU 7
block size: 32
KV heads / GQA group / head dim: 8 / 4 / 128
summary dtype: FP32
10 warmup + 50 measured iterations
256 MiB L2 flush before every timed call
```

Full precision/warp sweep:

```text
/tmp/cb_tc_selection_cold_l2_32k_128k.jsonl
```

Two 100-iteration repetitions of the best accuracy/speed mode:

```text
/tmp/cb_tc_selection_repeat_1.jsonl
/tmp/cb_tc_selection_repeat_2.jsonl
```

Best cold-L2 latency for each precision mode:

| context | existing kernel | IEEE Tensor Core | TF32x3 Tensor Core | TF32 Tensor Core |
|---:|---:|---:|---:|---:|
| 32K | 34.07 us | 34.94 us (0.98x) | 27.86 us (**1.22x**) | 26.77 us (1.27x) |
| 64K | 40.90 us | 53.50 us (0.76x) | 41.52 us (0.98x) | 40.19 us (1.02x) |
| 128K | 64.34 us | 82.55 us (0.78x) | 65.98 us (0.98x) | 64.30 us (1.00x) |

The repeated TF32x3 measurements were stable:

| context | round 1 | round 2 |
|---:|---:|---:|
| 32K | 1.223x | 1.215x |
| 64K | 0.976x | 0.970x |
| 128K | 0.975x | 0.975x |

Numerical sanity:

- IEEE: `s` max-abs about `6e-7`, `delta` max-abs about `2e-6`.
- TF32x3: `s` max-abs about `5e-7`, `delta` max-abs about `2.4e-6`.
- IEEE and TF32x3 had zero routing mismatches at reference-selected ratios
  `0.1%, 1%, 5%, 10%, 50%` in the 128K synthetic test.
- Plain TF32 had `s` max-abs about `5e-4`, `delta` max-abs about `2e-3`, and
  started changing routing at reference-quantile thresholds.  It is not an
  accuracy-preserving candidate even though the original formula is unchanged.

## Decision

Do **not** integrate this kernel into the production path.

The rewrite successfully moves the reductions to `tl.dot`, but it leaves the
dominant global-memory traffic unchanged: every layer still reads
`k_bar/k_max/k_min` for every block.  The padded MMA also performs extra work
because the useful GQA matrix is only 4 rows.  Tensor Core throughput helps at
the 32K latency floor, but 64K/128K are already dominated by streaming the
summary vectors, so the new arithmetic path is break-even or slower.

This result narrows the next optimization target: an exact-formulation speedup
must reduce or reuse summary bytes, not only accelerate their dot products.
Mixed summary storage and TMA already address part of that traffic; further
large gains require either a different summary/bound representation or a
multi-query execution scheme that amortizes the read without causing the
register spill seen in the existing multi-query prototype.
