# Best measured long-context configuration

This directory is the single reproducible entry point for the fastest
condition-block configuration measured so far on the RTX PRO 6000 Blackwell:

- fused Triton condition selection + hybrid page attention;
- native `block_size=64`, `eps=0.1`;
- exact mixed summaries plus approximate BF16 `k_bar` storage;
- post-prefill `StaticCache`;
- full decode-step CUDA graph;
- statistics disabled on the latency/serving path;
- TMA bounds disabled because it does not compose profitably with BF16
  `k_bar`.

The obsolete `CONDITION_BLOCK_COMPILE_SELECTION` and production
`--triton-chunk-blocks` knobs were removed during cleanup: CUDA fused selection
bypasses the historical PyTorch path, while chunk sizing only affects the
explicit legacy stage2 implementation.

Run LongBench quality with realized-budget statistics:

```bash
CUDA_VISIBLE_DEVICES=7 python -m \
  llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.best_long_context.run_longbench \
  --collect-stats --device cuda:0 --datasets narrativeqa gov_report \
  --limit 50 --max-new-tokens 128 --output-root /tmp/cb_best_quality
```

Run the optimized latency/serving path:

```bash
CUDA_VISIBLE_DEVICES=7 python -m \
  llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.best_long_context.run_longbench \
  --device cuda:0 --datasets gov_report --limit 10 --max-new-tokens 128 \
  --output-root /tmp/cb_best_fast
```

Run LongBench-v2 decode-only latency:

```bash
CUDA_VISIBLE_DEVICES=7 python -m \
  llama.script.analysis.condition_block_gen.longbench_v2_latency \
  --device cuda:0 --methods condition_block_triton --contexts 65536 131072 \
  --max-new-tokens 128 --samples 3 --warmup 1 --fixed-decode \
  --decode-only-timing --best-long-context-config \
  --output /tmp/cb_best_latency.jsonl
```

Run eager/CUDA-graph token parity:

```bash
CUDA_VISIBLE_DEVICES=7 python -m \
  llama.script.analysis.condition_block_gen.methods.condition_block_triton_impl.best_long_context.runtime_sanity \
  --device cuda:0 --contexts 32768 65536 131072 --max-new-tokens 8 \
  --output /tmp/cb_best_runtime_sanity.jsonl
```

The preset is optimized for latency at 128K.  BF16 `k_bar` is not bitwise
equivalent, and block 64 can select more pages than block 32 at the same
epsilon.  Quality and realized budget must therefore remain part of the
release gate.

## Full release sanity

All runs used GPU 7 exclusively.

Synthetic regressions:

- native block64 vs dense hybrid-attention reference: 4/4 routing cases have
  zero selected-mask mismatches; BF16 output max-abs is at most `4.88e-4`;
- FP32 vs BF16 `k_bar`, block32/64 and ordinary/TMA bounds: 4/4 cases have zero
  selected-mask mismatches; output max-abs is at most `2.44e-4`.

Runtime regression:

| target context | effective prompt | forced tokens | eager vs graph |
|---:|---:|---:|---:|
| 32K | 32,768 | 8 | exact |
| 64K | 65,536 | 8 | exact |
| 128K | 131,064 | 8 | exact |

Both paths reported stats disabled, and the graph path reported
`cuda_graph_decode=true`.

Paired LongBench quality against the default block32/FP32-summary path:

| dataset | samples | block32 score | preset score | delta | exact predictions | budget block32→preset |
|---|---:|---:|---:|---:|---:|---:|
| NarrativeQA | 50 | 27.15 | 28.46 | +1.31 | 29/50 | 0.188→0.227 |
| GovReport | 50 | 23.80 | 23.43 | -0.37 | 0/50 | 0.173→0.275 |
| MultiNews | 20 | 20.92 | 21.46 | +0.54 | 1/20 | 0.607→0.780 |

The score is stable, but this is not a matched-budget comparison.  Block64
selects more pages at the same epsilon; MultiNews in particular is already at
0.78 equivalent budget.  The preset is therefore a 128K latency preset, not a
universal replacement for block32.

Clean-tree CUDA-graph decode-only latency, 128 fixed tokens, one warmup plus
three measured samples:

| context | samples | mean | median | prior block32 baseline | speedup |
|---:|---|---:|---:|---:|---:|
| 64K | 2.0038 / 1.9925 / 1.9944 s | 1.9969 s | 1.9944 s | 2.0613 s | 1.032x |
| 128K | 2.0924 / 2.0823 / 2.0974 s | 2.0907 s | 2.0924 s | 2.1850 s | 1.045x |

The 64K comparison is cross-run and close to system noise.  The 128K result is
the intended operating point.  The same preset measured 2.0928s before code
cleanup, so removing dead paths caused no latency regression.

## Cleanup boundary

Removed:

- an unreferenced, older two-pass Triton selection kernel;
- the unreachable PyTorch `torch.compile` selection variant;
- misleading production CLI knobs for compiled selection and legacy-only
  chunk sizing.

Retained:

- dense and compact SDPA references used by numerical sanity;
- legacy Triton stage2 for explicit historical regression;
- isolated TMA, block64 and BF16-summary experiments with recorded results.
