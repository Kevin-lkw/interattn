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
5. [ ] ~~If quality holds at matched budget: Triton selection-stats kernel~~
   **NO-GO for the pure ball** (see verdict). The strict-bound low-IO candidate
   that survives the same evidence is the diagonal ellipsoid (`diag_ell`,
   `w` vector + `rho` scalar, 2/3 of the box read volume); a 5-sample PPL probe
   for it is included here (`ppl_diag_ell_ms.py`).

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

### Recommended next step

Run the full 20-sample diag_ell PPL sweep plus the LongBench probe (its
generation-side selection needs `w`/`rho` added to `gen_selection.py`, both
derivable from the prefix summaries already cached by the runner). If the
matched-budget parity holds, build the Triton selection-stats kernel reading
`k_bar` + `w` + `rho` (2 vectors + 1 scalar per block, ~1.5x selection-read
reduction vs box; at 128K that is roughly stats 1492 -> ~1000us/step by the
read-volume scaling that held for the mixed-summary experiments).

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
