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
- `SPEED_README.md` — dedicated speed analysis: diag_ell attention-phase and
  end-to-end speedups vs full attention, and the dominant cost inside each.

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

## Full LongBench eps sweep (16 datasets, production Triton path)

`run_sweep.py`: all 16 English+code LongBench datasets, full test sets
(3,750 samples/eps), eps grid {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1,
2.5, 5}, block 32, stats on. diag_ell ran through the production fused Triton
path with the diag_ell stats kernel; the box baseline was completed with the
same `condition_block_triton` method (the historical box sweep covered only
the 9 long-text datasets; the 7 short/code datasets were filled with
`run_condition_block_triton_eps.sh`). Both sides scored with the canonical
`longbench/eval.py` (its newline truncation for trec/triviaqa/samsum matters:
an ad-hoc scorer without it under-scores by tens of points).

Macro average over 16 datasets; budget = decode-only equivalent budget
(step-0 prefill excluded — the recorded per-sample budget is inflated by
1/n_steps amortization on short-output datasets and by degenerate long
outputs in the collapse regime, which also explains the apparent
budget-vs-eps non-monotonicity; pure selection budget is strictly monotone):

| eps | box score @ budget | diag_ell score @ budget |
|---:|---|---|
| 0.005 | 49.11 @ 0.705 | 49.33 @ 0.636 |
| 0.01 | 49.28 @ 0.586 | 49.24 @ 0.509 |
| 0.025 | 49.29 @ 0.415 | 49.15 @ 0.338 |
| 0.05 | 49.13 @ 0.294 | 49.00 @ 0.226 |
| 0.1 | 48.76 @ 0.193 | 48.45 @ 0.142 |
| 0.25 | 47.93 @ 0.105 | 47.11 @ 0.080 |
| 0.5 | 46.49 @ 0.071 | 44.27 @ 0.060 |
| 1.0 | 42.96 @ 0.056 | 39.62 @ 0.052 |

Interpolating box's macro curve at diag_ell's realized budgets, diag_ell is
within ±0.1 at high budget and +0.1..+0.4 ahead through the low-budget range
(clearly ahead at the collapse edge) — on the 16-dataset macro, diag_ell sits
on the Pareto frontier across the whole usable range while reading 1/3 fewer
selection bytes per block.

Per-category pattern (see `pareto_16ds_final.png` / `cmp16.json` in the
session scratchpad; regenerate from the raw jsonl):

- Retrieval-style QA (narrativeqa, qasper, multifieldqa_en, hotpotqa,
  2wikimqa, musique, triviaqa): diag_ell on the frontier — reaches the score
  plateau at 20-30% lower budget; box shows mid-budget collapse dips.
- Summarization (gov_report, qmsum, multi_news, samsum): box marginally
  ahead at matched budget in the low-budget band (<=0.5 Rouge-L); broad
  coverage favors the sign-adaptive box delta.
- Short/code tasks (trec, passage_count, passage_retrieval_en, lcc,
  repobench-p): the two are indistinguishable (both near the full-attention
  score down to very low budgets).

Raw outputs: `result/generate/diag_ell_sweep/` (diag_ell, + `eval.json`) and
`result/generate/Llama-3.1-8B-Instruct/longbench/` (box,
`eval_full_v2.json`).

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

### Official vs-full attention timing (decode attribution methodology)

`run_attribution.py` wraps `condition_block_decode_attribution.py` — the exact
methodology behind the impl README's box "vs full attention" table (clean
decode wall from an unprofiled run, per-kernel device-side times from a
separate profiled run; CUDA graph, post-prefill StaticCache, stats off,
GPU 7 exclusive, 128 fixed tokens). The box/full baselines reproduce the
recorded numbers within <1% (full decode 22.27/31.69/49.61 ms/step vs recorded
22.26/31.73/49.57; box attention 1686/2009/2890 us/step vs recorded
1691/2016/2897).

| context | full attn us/step | box attn | diag_ell attn | box vs full | **diag_ell vs full** | diag_ell vs box |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 2613 | 1686 | 1341 | 1.55x | **1.95x** | 1.26x |
| 64K | 7227 | 2009 | 1644 | 3.60x | **4.40x** | 1.22x |
| 128K | 13366 | 2890 | 2201 | 4.63x | **6.07x** | 1.31x |

Per-kernel at 128K: selection 1484 -> 1014 us/step (1.46x, matching the in-situ
profile), finalize 1320 -> 1102 (1.20x, fewer selected pages at the same eps).
Decode e2e vs full: 1.51x / 2.01x / 2.87x (box: 1.47x / 2.01x / 2.86x — the
attention win is GEMV-diluted, as always). Against the diag_ell read-volume
ceiling 1/(s + 1.5/B) ~ 16-20x, the realized 6.07x is ~1/3 — the remaining gap
is small-read bandwidth efficiency, unchanged from the box analysis.

### BF16 `w` storage (optional, still a strict bound)

Unlike BF16 `k_bar` (approximate), `w` can be stored in BF16 with a round-up
cast (`w * (1 + 2^-7)` before casting), which over-estimates delta by <=1% and
therefore keeps the bound valid. `bench_selection.py --w-dtype bfloat16`,
cold-L2:

| context | diag_ell FP32-w | diag_ell BF16-w | increment | vs box |
|---:|---:|---:|---:|---:|
| 32K | 25.6 us | 24.1 us | +6.3% | 1.44x |
| 64K | 31.7 us | 29.6 us | +7.0% | 1.39x |
| 128K | 48.2 us | 44.5 us | +8.4% | 1.43x |

The byte-count theory would allow +33%; the realized +6-8% shows the kernel is
increasingly latency-bound as reads shrink (same pattern as BF16 summaries on
the box kernel). Not yet wired into the production runner; adopting it needs
only the cheap selection-parity gate since delta changes by <=1% relative.

### BF16 `k_bar` storage (approximate center, delta still strict around it)

The last remaining stats-stream dtype lever (2026-07-24). Storage uses the
pre-existing core knob `CONDITION_BLOCK_K_BAR_DTYPE=bfloat16` (requires
`CONDITION_BLOCK_MIXED_SUMMARIES=1`); the diag_ell kernel, eager selection and
`diag_ell_stats` consume it unchanged (all loads upcast to FP32). Soundness
structure: `w` and `rho` are computed **from the stored BF16 center**, so
`delta = rho * ||q x w|| / sqrt(d)` remains a *strict* bound on the score
deviation around the stored center — the only approximation is the center
itself (`s` shifts by the k_bar rounding, so masses and scores shift by
~2^-8 relative; the mean-zero property behind the cosh term now holds only up
to that rounding). Incompatible with the legacy eager stage2 (core rejects
MIXED there), so all checks below run the fused production path.

Sanity (full stack `MIXED=1` + BF16 `k_bar` + BF16 `w`): `sanity.py` all
green (delta soundness zero violations against the stored center, box parity,
eps extremes); `runtime_sanity.py` eager vs CUDA-graph token parity exact at
32K/64K with the graph active.

Quality smoke (fused path, eps 0.1, scored with canonical `eval.py` against
the FP32 sweep reference on matched sample IDs — larger and
pipeline-identical, so not comparable to the 30-sample legacy-probe table
above):

| dataset | n | FP32 score @ budget | BF16-k_bar score @ budget | identical preds |
|---|---:|---|---|---:|
| narrativeqa (F1) | 200 | 29.11 @ 0.1731 | **29.36 @ 0.1733** | 176/200 |
| gov_report (Rouge-L) | 140 | 33.40 @ 0.1402 | **33.51 @ 0.1406** | 11/140 |

Score and budget are parity-or-better on both datasets (gov_report's low
identical-prediction count is expected: one token flip cascades through a
500-token summary; the score is what matters). Full PPL / 16-dataset gates
deliberately skipped this round (small sanity only); the PPL harness has an
env-gated BF16-k_bar cast (`ppl_condition.py`) if a deeper gate is wanted.

Selection-stats kernel, cold-L2 (`bench_selection.py --w-dtype bfloat16
--k-bar-dtype bfloat16`, same-session A/B):

| context | diag_ell BF16-w | + BF16 k_bar | increment |
|---:|---:|---:|---:|
| 32K | 24.0 us | 21.7 us | +9% |
| 64K | 29.4 us | 23.8 us | +19% |
| 128K | 44.7 us | 37.0 us | +17% |

Byte theory allows +32% at 128K (stats stream 26.4 -> 18.0 MB/layer); the
realized +17% is the familiar latency-bound halving, but a clearly larger
increment than the BF16-w step.

Official decode attribution (exclusive GPU 7, GEMV canary 10.8-11.0 ms/step
confirming no co-tenant; a first attempt that ran time-sliced against a
process that landed on GPU 7 mid-run was discarded):

| context | attention us/step (+BF16 w -> +BF16 k_bar) | vs full attn | e2e ms/step |
|---:|---|---:|---:|
| 32K | 1308 -> 1236 | 2.00x -> **2.11x** | 14.69 |
| 64K | 1540 -> 1381 | 4.69x -> **5.23x** | 15.42 |
| 128K | 2026 -> 1913 | 6.60x -> **6.99x** | 17.09 |

In-situ selection at 128K drops 976 -> 828 us/step (-15%, matching the
cold-L2 -17%); finalize/reduce are unchanged (999/86 us). e2e stays on the
GEMV wall as always. Data: `cb_attr_ball_diag_mixed_bf16w_bf16kbar.jsonl`.

## Selection-stage restructure toward full-attention/B (v2 -> v3)

Goal (2026-07-24): bring the selection-stats kernel to ~full attention /
block_size — at 128K that is 13366/32 = **418 us/step** (13.1 us/layer);
baseline after the dtype rounds was 828 us/step (efficiency ~41% of peak vs
flash's 71%, bytes already at the floor).

Step A — config sweep (20 configs, `SELECT_CHUNK` x `SELECT_WARPS`,
cold-L2): the existing kernel already sits at its optimum (c16_w2, 37.1 us at
128K). Bigger chunks catastrophically regress (chunk 64 -> 63 us, 128 ->
492 us): the broadcast-multiply-sum form materializes (G, B, D) FP32
intermediates, so registers scale with the tile and spill. Side finding: the
auto warp heuristic (8 below 1024 blocks) is stale after the dtype shrink —
`CONDITION_BLOCK_SELECT_WARPS=2` is ~20% faster at 32K.

Step B v2 (`triton_selection_v2.py`) — persistent grid, same math: best
config only *ties* v1 (36.1 vs 35.2 us; parity gated). Conclusion: the
bottleneck is the register wall of the compute formulation, not the grid.

Step B v3 (`triton_selection_v3.py`) — tensor-core rewrite: `s` and
`q^2.w^2` become `tl.dot` MMA (register footprint decoupled from the tile),
persistent span loop with software pipelining. Strictness is preserved: the
stored summary becomes `w2 = bf16(w*w)` — any positive weight keeps the
w-weighted Cauchy-Schwarz bound valid as long as `rho` is computed against
the *stored* value (it is) — and the delta scale carries a (1+2^-8)
inflation covering the BF16 rounding of q^2. Deltas differ from v1 by
~2^-8 in *both* directions (different valid weight), so borderline
selections can flip either way. Gates all green (`bench_selection_v3.py`):
soundness against dense exact deviations 0 violations at every context,
`s` matches v1 to 1.8e-7, delta matches its torch reference to 5.7e-6,
eager-vs-CUDA-graph token parity exact. Config landscape is flat
(28.3-30.3 us across 12 configs at 128K); default b32_s3_w4_p48.

In-situ decode attribution (three independent clean runs agreed to the
microsecond; GEMV canary 10.80/10.88/10.97 ms/step):

| context | selection us/step (v1 -> v3) | attention total | vs full attn |
|---:|---|---|---:|
| 32K | 444 -> **202** (2.20x) | 1237 -> 945 | 2.77x |
| 64K | 477 -> **332** (1.44x) | 1381 -> 1207 | 5.99x |
| 128K | 828 -> **516** (1.60x) | 1913 -> 1490 | **8.97x** |

Bonus: finalize dropped too (999 -> 887 us at 128K) because its inline
re-reduce of the selection partials shrank (BLOCK_SC 256 -> 64 via P=48).

Quality smoke (fused path, eps 0.1, canonical eval.py, matched IDs — same
protocol as the BF16-k_bar gate): narrativeqa 200 samples F1 29.42 @ budget
0.1727 (FP32 29.11 @ 0.1731, v1-BF16 29.36 @ 0.1733; 181/200 predictions
identical to v1), gov_report 140 samples Rouge-L 33.63 @ 0.1404 (FP32
33.40, v1 33.51). Parity-or-better at equal budget on both — the ±2^-8
selection flips are quality-neutral. PASS.

### Gap analysis vs the full/B target

- **128K: 516 vs 418 us/step = 1.23x — essentially reached.** Exact
  decomposition: 1.23 = 1.135 (bytes: 19.05 MB/layer streamed vs the
  16.78 MB full/B equivalent — the excess is the s/delta cache write
  2.10 MB, the structural price of the 3-kernel split, plus rho 0.13 MB and
  partials) x 1.076 (bandwidth efficiency 66% of peak vs flash's 71%).
  The kernel is no longer meaningfully less efficient than flash.
- **64K: 332 vs 226 = 1.47x** (efficiency 51%: 9.6 MB/layer is getting too
  small to amortize) and **32K: 202 vs 82 = 2.5x** (latency floor: 43% at
  4.9 MB/layer). The full/B target scales *down* linearly with context but
  fixed per-kernel costs do not — below ~64K the target is unreachable for
  any standalone selection kernel.
- Remaining lever: fusing stats+finalize would remove the s/delta round
  trip (-13.5% bytes), landing ~455 us/step at 128K (1.09x of target), and
  is also the only fix for the 32K floor. Beyond that, only hierarchical
  selection changes the byte count.

## Next: finalize restructure, then fusion (decided 2026-07-24)

Reading the production finalize kernel (`page_attention.py`): its heavy math
is already `tl.dot` MMA (no v3-style register wall), but its *grid* has the
same disease v1 stats had — (head, chunk) programs each touching ~13 KB
(v_bar tile 8 KB + s/delta 1 KB + inline partial re-reduce 4 KB) with no
loop to pipeline, plus the entire generated suffix processed serially by the
last chunk's single program. Measured ~30% of its byte bound (887 us/step at
128K). Plan: apply the v3 surgery — persistent span loop with pipelined
v_bar streaming, one partial re-reduce per program instead of per chunk,
suffix sliced across programs and merged in the out-reduce, FINALIZE_CHUNK
re-swept. Expected ~450-550 us/step at 128K (attention -> ~12-13x vs full).

Ordering rationale (finalize BEFORE fusion): (a) the prize is bigger —
finalize's efficiency deficit is worth ~350-450 us/step vs ~100-150 for
fusion's s/delta round trip + launch; fusing an inefficient finalize just
inherits its structure. (b) True fusion needs selection's global stats
before the routing decision — a grid-wide barrier, only feasible in the
cooperative persistent-program shape; a persistent finalize is therefore the
prerequisite, not the alternative. (c) The restructure is math-identical and
cheaply gated; fusion changes execution architecture and should merge two
already-clean components.

### Finalize v2 results (2026-07-24, `triton_finalize_v2.py`)

Implementation: persistent (head, P) grid, span loop over chunks,
selection-partial re-reduce once per program, suffix tiles distributed
round-robin (no serial tail), stage2 reduce over P partials. Parity gates
all green: `selected` sets bitwise identical to the production kernel at
every context including a stress eps that selects ALL 32768 blocks (full
page-loop exercise); output max err <= 3.6e-5 (FP32 merge-reorder noise, two
orders below BF16 output granularity); eager-vs-graph token parity exact.

**Key negative finding: persistence (span > 1) is wrong for finalize.**
Unlike stats (uniform work per block), finalize's selected-page work is
data-dependent and clustered on real prompts; consolidating chunks into
fewer programs serializes those page fetches and the in-situ numbers invert
the cold-L2 microbench (which selects almost no pages on synthetic data —
a measured methodology trap). In-situ finalize us/step:

| config | 32K | 64K | 128K |
|---|---:|---:|---:|
| production kernel | 688 | 796 | 887 |
| v2 P=48 (span 1/2/3) | 595 | 849 | 990 |
| **v2 P=128 (span=1, adopted)** | **596** | **693** | 903 |
| v2 P=64/BLOCK=64 | 719 | 844 | 1006 |

Adopted config `P=128/BLOCK=32` keeps the original chunk-level parallelism
(span=1) and banks the two unconditional wins — parallel suffix and
one-shot partial re-reduce: finalize **+15% at 32K and 64K**, tie at 128K
(903 vs 887; page-fetch latency dominates there and this surgery does not
touch it). Full v4 path (v3 stats + finalize v2, GEMV canary clean):

| context | attention us/step (sel/fin/red) | vs full attn | e2e ms/step |
|---:|---|---:|---:|
| 32K | 853 (203/596/55) | **3.06x** | 14.30 |
| 64K | 1104 (331/693/80) | **6.55x** | 15.16 |
| 128K | 1504 (515/903/86) | **8.88x** | 16.67 |

Remaining finalize headroom at 128K (~30% of its byte bound) is the
selected-page path itself: pages arrive via a serial data-dependent while
loop with one PAGE_SIZE tile per iteration. Ideas for a next round:
compact the selected list first so page fetches pipeline, split the
rep-stream and page-stream into specialized kernels, or take both into the
stats+finalize fusion.

### Two-stream finalize (2026-07-24, `triton_finalize_v3.py`) — adopted

The eps-knob diagnostic pinned the cost: at eps 1e9 (no pages) finalize is
only 197/270/401 us/step — the rep stream is fine; the selected-page path
adds ~400/420/500 us at eps 0.1 on ~37 us worth of bytes (real selections
cluster, the unluckiest per-chunk program walks 2-4 pages serially, the wave
waits), and explodes to ~2 ms at eps 0.01.

Split (`--selection diag_ell_split`, runtime_sanity `--fin-split`): kernel A
streams v_bar + condition + selection (uniform, persistent-safe) and stores
the group-shared mask; kernel B owns block b at program b % P_B, so adjacent
clustered pages land in distinct programs (critical path ~1 page), takes
suffix tiles round-robin, and both write disjoint slots of one partial array
merged by a single reduce. Deterministic, no atomics. Gates: selected sets
bitwise vs production incl. the all-blocks stress eps, output <= 6.8e-5,
eager-vs-graph token parity exact. In-situ (GEMV canary clean):

| context | finalize us/step (fin-v2 -> split) | attention total | vs full attn |
|---:|---|---|---:|
| 32K | 596 -> **360** (1.66x) | 689 (sel 203 / fin 360 / red 127) | **3.79x** |
| 64K | 693 -> **481** (1.44x) | 948 | **7.62x** |
| 128K | 903 -> **692** (1.30x) | 1364 | **9.80x** |

The wider reduce (P_A + P_B = 256 partials) costs +70 us at 128K, more than
covered. Real-tensor first-step bench agrees directionally (+3-7%; its cold
L2 and 1-token suffix dilute the page share — trust the in-situ numbers).
At eps 0.01 the split's benefit is far larger (the serial page walk was
~2 ms/step). Remaining finalize gap vs the no-page floor (692 vs ~401+B
overhead) is per-page latency in B (one 8 KB k + 8 KB v chain per page) —
further gains would need page prefetch across B's own hits or the fusion.

## Final summary (diag_ell round)

All gates passed; the strict-bound diag_ell selection is quality-neutral or
better at matched budget and reduces the attention phase by 1.32x in situ:

- Selection-stats kernel: 1.29-1.33x cold-L2 (theory for 3->2 vectors: 1.49x);
  1.46x in situ at 128K (~98% of the read-volume theory).
- Whole attention phase at 128K in situ: 3.15 -> 2.39 ms/step (**1.32x**,
  vs a fixed-selected-ratio theoretical 1.30x — slightly above because the
  sharper delta also selects fewer pages at the same eps).
- Official vs-full attention-phase speedup (decode-attribution methodology,
  baselines reproduced within <1%): **1.95x / 4.40x / 6.07x** at 32K/64K/128K
  (box: 1.55x / 3.60x / 4.63x), roughly 1/3 of the diag_ell read-volume
  ceiling 1/(s + 1.5/B) ~ 16-20x; the remaining gap is small-read bandwidth
  efficiency (~60% of peak), the same limitation documented for the box
  kernels.
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
