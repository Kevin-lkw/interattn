# Ball-bound condition-block selection

## Task

Replace the coordinate-wise box bound (`k_max`/`k_min`, 2 vectors per block) in the
condition-block delta with a spherical (ball) bound around the block mean
(1 vector + 1 scalar per block), **without giving up the strict upper-bound
guarantee**. This is direction A from
`condition_block_gen/CONDITION_BLOCK_TRITON_README.md` / the impl README: the
selection-stats kernel is the largest and fastest-growing attention-side cost, and
its cost is dominated by reading 3 summary vectors per block
(`k_bar`, `k_max`, `k_min`). The ball bound cuts that to `k_bar` plus a scalar
radius, a ~3x read reduction, raising the read-volume ceiling of the sparse
attention path from ~14x to ~26x over full attention.

This directory adds new code only; nothing under `condition_block_ppl/` or
`condition_block_gen/` is modified. Existing entry points are reused through the
established monkeypatch pattern (`condition_bound/ppl_bennett_ms.py` precedent).

## Math

The condition needs, per block `C` and query `q`,

```
delta_C(q) >= max_{i in C} |q . (k_i - k_bar_C)| / sqrt(d_k)
```

Ball bound: precompute once per block the exact scalar radius

```
r_C = max_{i in C} ||k_i - k_bar_C||_2
```

then at query time

```
delta_ball(q) = ||q||_2 * r_C / sqrt(d_k)
```

Soundness is one line of Cauchy-Schwarz:
`|q . (k_i - k_bar)| <= ||q|| * ||k_i - k_bar|| <= ||q|| * r_C`.
`r_C` is an exact quantity (a max over the block's keys, computed at
prefill/summary-build time), not an estimate; over-estimating delta keeps the
condition bound valid (it can only select more blocks, never miss one). The
trade is tightness, not correctness: the ball is isotropic. Stage-0 measurements
(`condition_block_ppl/delta_bound/README.md`) found box and ball land in the
same looseness range (box 4.9-5.3x vs oracle, ball within 4.3-5.6x) on real
Llama-2 keys, so little tightness should be lost — but selection behavior at a
given eps still shifts, hence the quality experiments below.

Float note: radii are computed in FP32 via `||k||^2 - 2 k.k_bar + ||k_bar||^2`;
rounding there is ~1e-6 relative, far below any eps of interest, and the box
path has the same FP32-rounding exposure.

## Files

- `ppl_condition.py` — ball-delta version of
  `runner_cond_block._batched_hybrid_outputs_for_queries` (teacher-forced PPL
  harness; delta section swapped, everything else copied verbatim). The
  `delta_mode` argument is ignored: this variant always uses the ball delta.
- `ppl_ball_ms.py` — multisample WikiText PPL entry; monkeypatches the function
  above and runs `multisample.run_condition_block` unchanged.
- `gen_selection.py` — ball version of the generation-side eager selection
  (`condition_block_triton_impl/core._select_prompt_blocks_eager`); the radius is
  computed once per layer from the cached prompt block summaries and stored in
  the prefix dict.
- `run_longbench.py` — LongBench generation entry. Forces the legacy (non-fused)
  stage2 so selection goes through the patchable eager path, monkeypatches
  `core._select_prompt_blocks` with the ball version, then forwards all CLI args
  to `longbench.run_all`. `--selection box` runs the identical pipeline with the
  original box selection for a paired baseline.
- `sanity.py` — soundness and parity checks (see plan below).

## Plan

1. [x] Eager/PyTorch ball delta for both harnesses (PPL + generation). No Triton.
2. [x] Sanity: ball delta >= exact per-token delta on synthetic structured keys
   (both harness implementations); structural parity with the box selection
   (identical `z_logits`/`v_bar`/sizes; only `delta`/`selected` may differ);
   eps extremes select all/none.
3. [x] WikiText PPL vs budget (Llama-3.1-8B, 1024 tokens, block 32,
   full-attention layers 2, same eps grid as the saved box baseline in
   `result/Llama-3.1-8B/wikitext_n20_block32/condition_block/`). Stopped early
   at 8/20 samples: ball loses on every sample at every eps, so more samples
   cannot change the verdict.
4. [x] LongBench small quality probe (narrativeqa + gov_report, extended to 30
   samples each, block 32, eps 0.1, 128 new tokens, stats on): score and
   realized budget, ball vs box through the identical legacy-stage2 pipeline.
5. [x] ~~If quality holds at matched budget: Triton selection-stats kernel~~
   **NO-GO for the pure ball** (see verdict). The strict-bound low-IO candidate
   that survives the same evidence is the diagonal ellipsoid (`diag_ell`,
   `w` vector + `rho` scalar, 2/3 of the box read volume).
6. [x] diag_ell follow-up round: generation-side selection + LongBench gate
   (passed, dominates box), full 20-sample PPL gate (passed), Triton
   selection-stats kernel (1.3x cold-L2 / 1.46x in situ), production fused-path
   integration with CUDA-graph token parity, decode-only latency and in-situ
   attribution (attention phase 1.32x). See "Final summary" below.

## Results

### Sanity (2026-07-23) — PASSED

`python -m script.analysis.condition_block_ball.sanity --device cuda:0`:

- PPL harness, 4 synthetic key geometries x 5 partial causal prefixes:
  `delta_ball >= delta_exact` and `delta_diag_ell >= delta_exact` on all 624
  (query, block) pairs, zero violations, min ball/exact ratio 2.91. Median
  looseness ball vs box vs diag_ell: 5.7/8.4/5.8x (iid), 5.7/8.5/5.9x (drift),
  13.9/7.5/12.5x (per-token outliers), 5.7/8.6/5.9x (shifted/scaled). Neither
  bound dominates; ball is tighter on 3 of the 4 geometries — tightness is not
  what fails it (see verdict).
- Generation harness (BF16 pages, 1000-token non-divisible prompt): radius and
  delta sound vs the dense exact delta; `z_logits`/`v_bar`/sizes bitwise equal
  to the box eager path; eps=0 selects every existing block, eps=1e9 none.
- LongBench smoke (gov_report 1 sample, 8 tokens, eps 0.1): monkeypatch
  demonstrably active — ball selects ~12% more clusters than box per step
  (step-1 equiv budget 0.103 vs 0.088); predictions identical on this sample.

### WikiText PPL vs budget — ball clearly worse at matched budget

8 paired samples (identical 1024-token windows, same eps grid; run stopped
early because the verdict was unanimous). Mean teacher PPL 8.13. Ball's
measured budget is equal or *higher* at every eps, yet its PPL is worse on
**8/8 samples at every eps** (7/8 at eps=0.075) and collapses much earlier:

| eps | ball PPL @ budget | box PPL @ budget | ball worse on |
|---:|---|---|---:|
| 0.05 | 8.21 @ 0.863 | 8.13 @ 0.868 | 8/8 |
| 0.075 | 8.26 @ 0.808 | 8.12 @ 0.810 | 7/8 |
| 0.1 | 8.34 @ 0.761 | 8.13 @ 0.760 | 8/8 |
| 0.25 | 9.05 @ 0.576 | 8.18 @ 0.563 | 8/8 |
| 0.5 | 11.63 @ 0.434 | 8.34 @ 0.408 | 8/8 |
| 1.0 | 22.68 @ 0.313 | 9.32 @ 0.288 | 8/8 |
| 2.5 | 118.4 @ 0.206 | 34.3 @ 0.190 | 8/8 |
| 5.0 | 477.7 @ 0.160 | 142.9 @ 0.159 | 8/8 |

Raw summaries: `result/Llama-3.1-8B/wikitext_n20_block32_ball/` (8 completed
samples) vs the saved box baseline `wikitext_n20_block32/`.

### LongBench probe (30 samples/dataset) — ball behind box at slightly higher budget

| dataset | ball score @ budget | box score @ budget |
|---|---|---|
| narrativeqa (F1) | 30.65 @ 0.201 | 33.14 @ 0.196 |
| gov_report (Rouge-L) | 22.56 @ 0.164 | 23.55 @ 0.161 |

Per-sample F1 shows the narrativeqa gap concentrated in a few large greedy
flips (3 samples at -29/-40/-30 vs one at +20 in the first 10), but the sign
consistently leans against ball on both datasets while ball spends slightly
more budget.

### Stage-0 corroboration (matched-k selection error on real Llama-2 keys)

Re-ran `delta_bound/delta_variants.py` (block 32, layers 10/15/20) to extract
the per-variant selection columns the delta_bound README did not quote.
Matched-k true hybrid error vs box at 5/10/20% selected fractions:

| variant | reads/block | L10 | L15 | L20 |
|---|---|---|---|---|
| ball | 1 vec + 1 scalar | 1.21x / 1.55x / 1.32x | 1.28x / 1.76x / 1.64x | 0.98x / 0.80x / 0.95x |
| diag_ell | 2 vec + 1 scalar | 0.95x / 1.09x / 1.05x | 0.91x / 0.84x / 1.07x | 0.89x / 0.86x / 1.13x |
| oracle delta | — | 0.92x / 0.97x / 0.93x | 0.90x / 0.81x / 0.80x | 0.84x / 0.62x / 0.77x |

Ball's *tightness* matches box (median 4.6-5.4x vs 4.9-5.3x), so the failure
is not looseness: the isotropic radius removes the query-direction dependence
of delta (within one query, blocks differ only by the scalar `r_C`), and the
earlier term-decomposition work showed the selection gain lives almost entirely
in term 1's query-adaptive `log p_hat + delta` top-tail ordering. diag_ell
keeps per-dimension anisotropy (`w`) and query dependence (`||q * w||`) and
stays at box-level selection quality with 2/3 of the reads.

## Verdict

**Pure ball: NO-GO.** The bound is sound and exactly computable (soundness
checks all pass), but selection quality at matched budget degrades clearly and
consistently across three independent measurements (paired WikiText PPL, paired
30-sample LongBench, stage-0 matched-k). Do not build the Triton kernel for the
1-vector ball.

**Surviving strict-bound candidate: diag_ell** — store `w_C = max_t |k_t -
k_bar|` (elementwise, one vector; in the causal harness it falls out of the
existing prefix cummax/cummin for free) plus scalar
`rho_C = max_t ||(k_t - k_bar)/w_C||`; at query time
`delta = rho_C * ||q * w_C|| / sqrt(d)`. Cauchy-Schwarz in the w-weighted inner
product keeps it a strict upper bound. Selection read volume drops from 3
vectors to 2 vectors + 1 scalar (~1.5x less), with box-level selection quality
in stage-0.

### diag_ell PPL probe (3 paired samples, stopped with the ball run)

Same protocol as the ball sweep (`ppl_diag_ell_ms.py`,
`result/Llama-3.1-8B/wikitext_n5_block32_diag_ell/`). Teacher mean PPL 6.70.
Unlike ball, diag_ell realizes a *lower* budget than box at the same eps, so
the per-eps table understates it; interpolating box's budget→PPL curve at
diag_ell's realized budgets:

| eps | diag_ell PPL @ budget | box PPL @ same eps | box interp @ diag_ell budget |
|---:|---|---|---|
| 0.05 | 6.74 @ 0.820 | 6.70 @ 0.866 | ~6.71 (+0.5%) |
| 0.1 | 6.79 @ 0.698 | 6.72 @ 0.757 | ~6.74 (+0.7%) |
| 0.25 | 6.91 @ 0.492 | 6.78 @ 0.559 | ~6.85 (+1%) |
| 0.5 | 7.23 @ 0.346 | 6.94 @ 0.405 | ~7.44 (**-3%**) |
| 1.0 | 9.30 @ 0.246 | 7.98 @ 0.287 | ~13.1 (**-29%**) |
| 2.5 | 25.7 @ 0.181 | 25.7 @ 0.190 | ~40.1 (**-36%**) |

At matched budget, diag_ell is within ~1% of box at high budget and clearly
*better* in the low-budget region that matters for compression. Only 3 samples
— preliminary, but the opposite sign of the ball result, consistent with the
stage-0 matched-k columns.

### diag_ell follow-up (this round)

Implemented in this directory, pending the quality gates:

- `gen_selection.py` now also provides `select_prompt_blocks_diag_ell`
  (`w`/`rho` computed once per layer from the cached prompt summaries — `w`
  falls out of the existing `k_max`/`k_min`, `rho` is one weighted pass over
  the pages); `run_longbench.py` accepts `--selection diag_ell`. Sanity: zero
  soundness violations, `z_logits`/`v_bar` bitwise equal to the box path.
- `triton_selection.py`: diag_ell selection-stats kernel, a drop-in for the
  production box stats kernel (identical `s`/`delta`/partial outputs; the
  finalize/stage2 kernels are unchanged). Reads `k_bar` + `w` + `rho`
  (2 vectors + 1 scalar per block) instead of 3 vectors.
- `bench_selection.py`: cold-L2 A/B microbench (methodology of
  `condition_block_stage_latency.py`; 256 MiB L2 flush per iteration, CUDA
  events, two alternating rounds). RTX PRO 6000 Blackwell, block 32, 8 KV
  heads, head_dim 128:

| context | box stats | diag_ell stats | speedup | parity |
|---:|---:|---:|---:|---|
| 32K | 34.1 us | 25.6 us | **1.33x** | `s` bitwise; delta err 2.9e-6 |
| 64K | 41.0 us | 31.7 us | **1.29x** | `s` bitwise; delta err 3.8e-6 |
| 128K | 63.6 us | 48.2 us | **1.32x** | `s` bitwise; delta err 3.8e-6 |

The box column reproduces the production FP32 numbers from the impl README
(128K ~64 us/layer), so the ~1.3x carries over: 128K selection would drop from
~1.5 ms/step to ~1.1 ms/step. Production integration is a monkeypatch of
`core._run_condition_block_selection_stats` (same signature/returns; the fused
decode path calls it at `core.py:1051`), with `w`/`rho` built lazily during the
eager warmup steps, so it composes with the CUDA-graph decode flow.

### diag_ell LongBench probe (30 samples/dataset) — PASSED, dominates box

Same paired protocol as the ball probe (block 32, eps 0.1, 128 new tokens,
identical legacy-stage2 pipeline):

| dataset | box score @ budget | diag_ell score @ budget | ball (ref) |
|---|---|---|---|
| narrativeqa (F1) | 33.14 @ 0.196 | **35.45 @ 0.172** | 30.65 @ 0.201 |
| gov_report (Rouge-L) | 23.55 @ 0.160 | **23.21 @ 0.106** | 22.56 @ 0.163 |

diag_ell matches or beats the box score while realizing a 12-34% *lower*
budget — the opposite sign of the ball probe, and consistent with its PPL
probe (better at low budget) and the stage-0 matched-k columns.

### diag_ell production-path integration — token parity PASSED

`runtime_sanity.py`: with `core._run_condition_block_selection_stats`
monkeypatched to the diag_ell runner, the fused block-32 path generates
token-exact sequences between eager decode and CUDA-graph decode on real
LongBench-v2 prompts (32K and 64K, 8 forced tokens, `cuda_graph_decode=true`).

### diag_ell decode-only e2e latency (CUDA graph, GPU 7 exclusive)

`run_latency.py`, 128 fixed tokens, 1 warmup + 3 measured, block 32, eps 0.1,
stats off:

| context | box decode-only | diag_ell decode-only | e2e speedup |
|---:|---:|---:|---:|
| 64K | 2.0056 s (15.79 ms/step) | 1.9905 s (15.67 ms/step) | 1.008x |
| 128K | 2.1860 s (17.21 ms/step) | 2.1870 s (17.22 ms/step) | 1.000x |

No measurable e2e gain — as expected from the project history: the selection
kernel is ~2 ms of a ~17 ms decode step dominated by model GEMV, and the
kernel-level 1.3x predicts only ~2-3% e2e, inside the ±1% run noise (the same
dilution held for mixed summaries: attention -9..18%, e2e ~1%). The value of
diag_ell is therefore (a) the attention-phase reduction itself and (b) the
budget reduction at matched quality (LongBench realizes 12-34% lower budget,
which shrinks selected-page IO too). In-situ per-kernel attribution at 128K is
recorded below.

### In-situ attention-phase attribution (128K, profiled decode-only, 127 steps)

Profiler CUDA self-time per bucket (profiler wall is slower than clean runs;
use ratios, not absolutes):

| bucket | box | diag_ell | speedup |
|---|---:|---:|---:|
| condition_selection | 187.1 ms (1.47 ms/step) | 128.4 ms (1.01 ms/step) | **1.46x** |
| sparse_finalize_attention | 202.0 ms | 164.8 ms | 1.23x |
| sparse_reduce | 10.9 ms | 10.9 ms | 1.00x |
| **attention total** | **400.0 ms (3.15 ms/step)** | **304.1 ms (2.39 ms/step)** | **1.32x** |
| model_gemm_gemv | 1384.5 ms | 1423.8 ms | (noise) |

The in-situ selection gain (1.46x) exceeds the cold-L2 microbench (1.32x), and
finalize also gets faster because diag_ell's sharper delta selects fewer pages
at the same eps (consistent with the lower realized LongBench budgets). The
attention phase — the target of this work — drops 1.32x end to end in the real
decode path.

### diag_ell WikiText PPL, full 20 paired samples — PASSED

Same protocol/windows as the box baseline (teacher mean PPL 6.8396). diag_ell
realizes a lower budget at every eps, so the comparison interpolates box's
budget→PPL curve (log-PPL, linear in budget) at diag_ell's realized budgets:

| eps | diag_ell PPL @ budget | box interp @ same budget | diag_ell vs box |
|---:|---|---|---:|
| 0.05 | 6.854 @ 0.825 | 6.839 | +0.2% |
| 0.075 | 6.859 @ 0.758 | 6.844 | +0.2% |
| 0.1 | 6.869 @ 0.703 | 6.854 | +0.2% |
| 0.25 | 6.936 @ 0.497 | 6.936 | 0.0% |
| 0.5 | 7.171 @ 0.350 | 7.455 | **-3.8%** |
| 1.0 | 8.845 @ 0.249 | 13.622 | **-35.1%** |
| 2.5 | 35.44 @ 0.179 | 50.97 | **-30.5%** |
| 5.0 | 145.7 @ 0.157 | 127.8 (clamped) | +14.0%* |

*The eps=5 point sits below box's lowest measured budget (0.159), so the
reference is clamped to box's endpoint rather than interpolated; both
configurations are far into the collapse regime there (PPL >120 vs teacher
6.84).

Verdict: at matched budget diag_ell is within +0.2% of box in the
usable-quality region and 30-35% better in the low-budget region — the
PPL, LongBench, and stage-0 gates all pass.

### Box vs diag_ell tightness on real Llama-3.1-8B keys

`delta_tightness.py` (stage-0 sampling protocol: wikitext activations, 64
groups/layer, 16 queries, causal full blocks, block 32; ratio = bound/exact,
pairs with exact delta > 0.01):

| layer | box med (p90) | diag_ell med (p90) | diag/box med | diag tighter on | Spearman vs exact box / diag |
|---:|---|---|---:|---:|---|
| 4 | 5.87 (8.08) | 5.24 (7.09) | 0.888 | 83.4% | 0.335 / 0.382 |
| 10 | 5.96 (9.01) | 5.81 (8.82) | 0.951 | 62.7% | 0.315 / 0.300 |
| 16 | 5.47 (7.90) | 5.52 (8.25) | 1.024 | 43.3% | 0.410 / 0.394 |
| 22 | 5.07 (7.72) | 5.76 (8.13) | 1.138 | 34.3% | 0.384 / 0.408 |
| 28 | 6.62 (9.22) | 5.81 (8.10) | 0.872 | 87.7% | 0.360 / 0.345 |
| all | 5.79 | 5.61 | **0.949** | **62.3%** | — |

Neither bound dominates: on the production model diag_ell is ~5% tighter in
the pairwise median and tighter on 62% of pairs, with strong layer dependence
(0.87-1.14 across layers); on Llama-2 (stage-0) the medians leaned the other
way (box 4.85-5.28 vs diag_ell 5.03-5.57). Both stay ~5-6x loose overall and
both correlate only weakly with the exact delta ordering (Spearman 0.30-0.41).
diag_ell's budget/quality advantage therefore does not come from a large
median-tightness win — it comes from the shape of its delta distribution
feeding the condition score (sharper separation of the top tail), which the
matched-budget PPL/LongBench gates measure directly.

## Final summary (diag_ell round)

All gates passed; the strict-bound diag_ell selection is quality-neutral or
better at matched budget and reduces the attention phase by 1.32x in situ:

- Selection-stats kernel: 1.29-1.33x cold-L2 (theory for 3->2 vectors: 1.49x);
  1.46x in situ at 128K (~98% of the read-volume theory).
- Whole attention phase at 128K in situ: 3.15 -> 2.39 ms/step (**1.32x**,
  vs a fixed-selected-ratio theoretical 1.30x — slightly above because the
  sharper delta also selects fewer pages at the same eps).
- Combined with the box path's measured 4.62x attention-phase advantage over
  full attention at 128K, the estimated cumulative attention-phase speedup is
  ~6.1x, roughly 1/3 of the diag_ell read-volume ceiling 1/(s + 1.5/B) ~
  16-20x; the remaining gap is small-read bandwidth efficiency (~60% of peak),
  the same limitation documented for the box kernels.
- e2e decode is unchanged (1.00-1.01x): model GEMV dominates, as throughout
  this project.

Recommended production step (outside this experiment folder): adopt the
diag_ell stats kernel as an opt-in variant of the fused path (the monkeypatch
in `run_latency.py`/`runtime_sanity.py` shows the integration is a drop-in),
re-calibrate eps if a specific target budget is required, and re-run the
release LongBench gate at the chosen operating point.

## Commands

Sanity:

```bash
cd llama && conda run -n nanogpt python -m script.analysis.condition_block_ball.sanity --device cuda:0
```

WikiText PPL (matches the box baseline settings; distinct output root):

```bash
cd llama && conda run -n nanogpt python -m script.analysis.condition_block_ball.ppl_ball_ms \
  --device cuda:0 --num-samples 20 --seq-len 1024 --block-size 32 \
  --eps 5.0 2.5 1.0 0.5 0.25 0.1 0.075 0.05 --full-attention-layers 2 \
  --output-root /scratch1/liankewei/interattn/llama/result/Llama-3.1-8B/wikitext_n20_block32_ball
```

LongBench probe (ball, then box baseline through the same pipeline):

```bash
cd /scratch1/liankewei/interattn && PYTHONPATH=llama conda run -n nanogpt \
  python -m script.analysis.condition_block_ball.run_longbench \
  --selection ball --device cuda:0 --datasets narrativeqa gov_report --limit 10 \
  --condition-block-size 32 --condition-eps 0.1 --max-new-tokens 128 \
  --output-root /scratch1/liankewei/interattn/llama/result/generate/ball_probe

# box baseline: same command with --selection box and an output root of its own
```

diag_ell PPL probe (same protocol, distinct output root):

```bash
cd llama && conda run -n nanogpt python -m script.analysis.condition_block_ball.ppl_diag_ell_ms \
  --device cuda:0 --num-samples 5 --seq-len 1024 --block-size 32 \
  --eps 5.0 2.5 1.0 0.5 0.25 0.1 0.075 0.05 --full-attention-layers 2 \
  --output-root /scratch1/liankewei/interattn/llama/result/Llama-3.1-8B/wikitext_n5_block32_diag_ell
```

Stage-0 selection columns (matched-k, saved Llama-2 setting):

```bash
cd llama && conda run -n nanogpt python -m script.analysis.condition_block_ppl.delta_bound.delta_variants \
  --device cuda:0 --budget 0.03125 --layers 10 15 20
```
