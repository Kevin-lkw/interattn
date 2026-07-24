# diag_ell speed analysis: attention phase and end-to-end

Question answered here: how much faster is the diag_ell condition-block path
than full attention — (a) in the attention phase, (b) end to end — and what is
the current dominant cost inside each.

## Methodology

All numbers come from the official decode-attribution runs
(`run_attribution.py`, wrapping `condition_block_decode_attribution.py`):
clean decode wall from an unprofiled run, per-kernel device-side times from a
separate profiled run. Llama-3.1-8B-Instruct, BF16, batch 1, block 32,
eps 0.1, stats off, CUDA-graph decode, post-prefill StaticCache, 128 fixed
tokens, RTX PRO 6000 Blackwell, exclusive GPU. The `full` baseline is HF
`generate` with DynamicCache (the standard baseline used in all prior
condition-block measurements; its own cache-append cost is called out below).
Box/full baselines reproduce the numbers recorded in
`condition_block_triton_impl/README.md` within <1%.

Raw data: `llama/result/generate/cb_attr_ball_box.jsonl` (full + box),
`cb_attr_ball_diag.jsonl` (diag_ell).

## 1. Attention-phase speedup vs full attention

Attention phase = `flash_sdpa_attention` for full; `condition_selection` +
`sparse_finalize_attention` + `sparse_reduce` for the condition-block paths.

| context | full | box | diag_ell | box vs full | **diag_ell vs full** |
|---:|---:|---:|---:|---:|---:|
| 32K | 2613 us/step | 1686 | 1341 | 1.55x | **1.95x** |
| 64K | 7227 us/step | 2009 | 1644 | 3.60x | **4.40x** |
| 128K | 13366 us/step | 2890 | 2201 | 4.63x | **6.07x** |

Relative to box, diag_ell's gain (1.22-1.31x) realizes ~100% of the
3-vector -> 2-vector summary-read theory.

### Ceilings

| context | theoretical ceiling | reasonable ceiling |
|---:|---:|---:|
| 32K | 16.4x | ~2.6x |
| 64K | 18.6x | ~6.1x |
| 128K | 19.7x | **~8.4x** |

**Theoretical ceiling** = pure read-volume bound `1/(s + 1.5/B)` at the
production selected ratios (s ≈ 1.4% / 0.7% / 0.4%): summaries counted as
2 vectors + 1 scalar per block for selection plus 1 vector for `v_bar`,
against 2 vectors per token for full attention. It assumes summaries cost
the same bytes as BF16 K/V, zero intermediate traffic, and flash-level
kernel efficiency — none of which hold, so it is an upper bound only.

**Reasonable ceiling** = today's measured speedup x the byte reduction still
available *without weakening the bound or restructuring kernels*:

```
reasonable = measured x (bytes_today / bytes_after_BF16_v_bar_and_w)
```

where the per-layer byte accounting covers every stream (summaries, rho,
selected pages, suffix, s/delta caches, softmax partials), and the remaining
sound dtype moves are BF16 `v_bar` (exactly equivalent in the mixed layout)
and round-up BF16 `w` (over-estimates delta by <=1%, bound stays strict).
E.g. at 128K: bytes 59.4 -> 42.6 MB/layer, so 6.07x x 59.4/42.6 ≈ 8.4x.
Holding each context's measured bandwidth efficiency fixed is what makes
this "reasonable": the shrinking reads are latency-bound (32K sits on the
3-kernel launch floor, which is why its ceiling is barely above today), and
pushing efficiency toward flash level would need persistent-kernel
restructuring previously assessed as low-ROI. Accepting approximate BF16
`k_bar` on top would stretch 128K to ~10.5x.

## 2. End-to-end decode speedup vs full attention

Clean decode wall per step (128 fixed tokens):

| context | full (HF DynamicCache) | diag_ell | **e2e speedup** | box (ref) |
|---:|---:|---:|---:|---:|
| 32K | 22.27 ms/step | 14.75 ms/step | **1.51x** | 14.96 (1.49x) |
| 64K | 31.69 ms/step | 15.73 ms/step | **2.01x** | 15.70 (2.02x) |
| 128K | 49.61 ms/step | 17.26 ms/step | **2.87x** | 17.31 (2.87x) |

Two caveats:

- A large share of the full baseline's cost at long context is its own
  DynamicCache `torch.cat` (see breakdown below: 22.4 of 49.6 ms/step at
  128K). Against an idealized full baseline with in-place cache writes
  (17.4 / 21.5 / 27.6 ms/step, recorded in the impl README), the e2e speedup
  is ~1.0x / 1.37x / 1.60x.
- diag_ell ≈ box end to end (1.00-1.01x): the attention-phase win is diluted
  by the model GEMV wall, exactly as every prior attention-side optimization.

## 3. What dominates the attention phase now (diag_ell)

Per-kernel attention breakdown, us/step:

| context | selection stats | finalize attention | reduce | attention total |
|---:|---:|---:|---:|---:|
| 32K | 542 | **744** | 55 | 1341 |
| 64K | 680 | **884** | 80 | 1644 |
| 128K | 1014 | **1102** | 86 | 2201 |

**The largest attention component is now the finalize kernel**, not selection
(selection was the largest under box: 1484 vs 1320 at 128K; diag_ell cut
selection by 1.46x, and finalize by 1.20x through fewer selected pages, which
flipped the order). Finalize's cost is dominated by streaming the
representative values (`v_bar`, one vector per block) plus the selected token
pages and the generated suffix, and writing per-chunk softmax partials.

### Per-kernel distance from the byte-bound optimum

Byte-exact lower bound (all streams, 1.8 TB/s peak) vs measured, per layer:

| context | selection: ideal / measured (eff) | finalize: ideal / measured (eff) |
|---:|---|---|
| 32K | 4.8 / 16.9 us (29%) | 4.0 / 23.3 us (**17%**) |
| 64K | 9.7 / 21.3 us (46%) | 6.7 / 27.6 us (**24%**) |
| 128K | 19.4 / 31.7 us (61%) | 12.4 / 34.4 us (**36%**) |

Neither kernel is at its optimum, but the deficit is asymmetric: selection
runs at 61% of its byte-bound at 128K (comparable to the tuned box stats
kernel; latency-bound at 32K), while **finalize runs at only 17-36%** — even
with selection free, finalize alone would still be ~2.8x above its byte
bound at 128K. The reasons are structural: each (head, chunk) program
touches only a 32-block v_bar tile plus rarely-selected pages (little work
to hide memory latency), the online softmax is a serial chain within the
program, and the entire generated suffix is processed serially by the last
chunk's program. So the *largest single kernel-engineering lever left in the
attention phase is finalize bandwidth efficiency (~2.4x at 128K if it
reached flash-level ~85%)*, ahead of any remaining dtype trick; it would
require restructuring (wider tiles / persistent programs), which is the same
class of work previously assessed as low-ROI for the stats kernel — but the
headroom here is larger than it was there.

### Measured dtype results (2026-07-24)

Both sound dtype levers were implemented and measured end to end
(attribution methodology, `CONDITION_BLOCK_MIXED_SUMMARIES=1` for exact BF16
`v_bar`, `CONDITION_BLOCK_BALL_W_DTYPE=bfloat16` for round-up BF16 `w` with
`rho` recomputed from the stored `w`, so the bound stays strict for any
positive weight):

| context | attention us/step FP32 -> +mixed -> +BF16 w | vs full attn | e2e |
|---:|---|---:|---:|
| 32K | 1341 -> 1336 -> 1308 | 1.95x -> **2.00x** | unchanged |
| 64K | 1644 -> 1591 -> 1540 | 4.40x -> **4.69x** | unchanged |
| 128K | 2201 -> 2064 -> 2026 | 6.07x -> **6.60x** | unchanged |

Sanity, all green: kernel soundness zero violations with BF16 `w`;
mixed-summary predictions are **exactly identical** to FP32 (3/3 smoke,
budgets equal to 4 decimals — the exact-equivalence argument holds for
diag_ell); mixed+BF16-w also produced identical predictions on the smoke (no
boundary blocks flipped); eager vs CUDA-graph token parity exact.

Note the realized gains (finalize -12.7%, stats -4..8% at 128K) are well
below the byte-count reduction (-38% finalize stream, -24% stats stream):
shrinking reads push the kernels further into the latency-bound regime, so
the earlier fixed-efficiency "reasonable ceiling" of ~8.4x was optimistic —
**with all sound dtype levers applied, the measured ceiling is ~6.6x at
128K**, and essentially all remaining headroom (finalize now runs at ~26% of
its new byte bound) is kernel-structural, not dtype.

### BF16 `k_bar` round (2026-07-24)

The last stats-stream dtype lever, measured and quality-gated (see the main
README's BF16 `k_bar` section for the full record). Config:
`CONDITION_BLOCK_K_BAR_DTYPE=bfloat16` on top of mixed summaries + BF16 `w`.
The stored center is approximate (~2^-8 relative) but `w`/`rho` are computed
from it, so delta stays a strict bound around the stored center.

- Quality: fused-path LongBench smoke vs the FP32 sweep reference on matched
  IDs — narrativeqa 200 samples 29.11 -> 29.36 F1, gov_report 140 samples
  33.40 -> 33.51 Rouge-L, budgets within +0.03% relative; sanity + CUDA-graph
  token parity all green. PASS.
- Selection kernel, cold-L2: 24.0 -> 21.7 / 29.4 -> 23.8 / 44.7 -> 37.0 us at
  32K/64K/128K (**+9% / +19% / +17%**; byte theory +32% at 128K — stats
  stream 26.4 -> 18.0 MB/layer, the same latency-bound halving as every
  dtype step). The summary read cost per block drops to 516 B ~ 1 BF16
  token (K+V), i.e. the stats byte floor now sits essentially at the
  "full-attention / block-size" level (see the selection-vs-full/B analysis).
- In-situ attribution (exclusive GPU, GEMV canary 10.8-11.0 ms/step):
  attention 1308 -> 1236 / 1540 -> 1381 / 2026 -> 1913 us/step at
  32K/64K/128K, i.e. **2.11x / 5.23x / 6.99x vs full** (from 2.00/4.69/6.60).
  Selection at 128K: 976 -> 828 us/step (-15%, matching cold-L2); finalize
  unchanged (999 us) and is now clearly the dominant attention kernel at
  every context. e2e 14.69/15.42/17.09 ms/step — flat, GEMV wall.
  The *sound*-dtype ceiling stays ~6.6x (BF16 k_bar is approximate); with
  the approximate center accepted, the measured dtype-exhausted ceiling at
  128K moves to **~7.0x**.

### v3 tensor-core selection kernel (2026-07-24)

The selection-stats kernel was restructured to `tl.dot` MMA with a
persistent pipelined loop (`triton_selection_v3.py`; full record and gap
analysis in the main README). Strict bound kept (stored `w2` weight + rho
from it + (1+2^-8) inflation). Clean in-situ attribution (GEMV canary
10.8-11.0, three agreeing runs):

| context | selection us/step | attention us/step | vs full attn | e2e ms/step |
|---:|---:|---:|---:|---:|
| 32K | 444 -> **202** | 945 | **2.77x** | 15.67 |
| 64K | 477 -> **332** | 1207 | **5.99x** | 15.32 |
| 128K | 828 -> **516** | 1490 | **8.97x** | 16.72 |

Selection now runs at 66% of peak bandwidth at 128K (flash: 71%) — 1.23x
above the full-attention/B time target, decomposed as 1.135x bytes (the
s/delta cache round trip of the 3-kernel split) x 1.076x efficiency. The
earlier "dtype-exhausted ~7.0x" attention ceiling was a statement about
dtype levers only; kernel restructuring moved the measured phase to 8.97x.

### Finalize v2 (persistent grid, 2026-07-24)

Applied the v3 surgery to finalize (`triton_finalize_v2.py`; full record in
the main README). Outcome: span>1 persistence HURTS finalize in situ
(selected-page work is data-dependent and clustered — a cold-L2 blind
spot); the adopted P=128/span=1 config banks the unconditional wins only
(parallel suffix + one-shot partial re-reduce). Current totals (v4 = v3
stats + finalize v2, clean attribution):

| context | attention us/step | vs full attn | e2e ms/step |
|---:|---:|---:|---:|
| 32K | 853 | **3.06x** | 14.30 |
| 64K | 1104 | **6.55x** | 15.16 |
| 128K | 1504 | **8.88x** | 16.67 |

### Two-stream finalize (2026-07-24) — current best

Splitting finalize into a uniform rep kernel + a page-parallel exact kernel
(cluster-busting b % P_B ownership; full record in the main README) fixed
the serial page walk the eps-knob diagnostic exposed (pages: ~500 us/step
at 128K on ~37 us of bytes). Current totals (v3 stats + split finalize,
clean attribution):

| context | attention us/step | vs full attn | e2e ms/step |
|---:|---:|---:|---:|
| 32K | 689 | **3.79x** | 14.33 |
| 64K | 948 | **7.62x** | 15.10 |
| 128K | 1364 | **9.80x** | 16.53 |

Remaining attention-side levers:

1. Per-page latency inside the page kernel (one 8 KB + 8 KB chain per hit):
   prefetch across a program's own hits, or absorb into fusion.
2. Stats+finalize fusion: removes the s/delta round trip and the 32K launch
   floor; both component kernels are now persistent-shaped and clean.
3. block 64 (halves summary count; needs a fresh quality/budget gate).

## 4. What dominates end to end

Full per-category breakdown, us/step (busy = GPU busy; wall includes ~5%
host/launch):

| bucket | full 128K | diag_ell 32K | diag_ell 64K | diag_ell 128K |
|---|---:|---:|---:|---:|
| model GEMM/GEMV | 10637 | 10776 (73%) | 10836 (69%) | 10957 (63%) |
| attention phase | 13366 | 1341 (9.6%) | 1644 (11.1%) | 2202 (13.6%) |
| KV cache copy/index | **22367** | 617 | 750 | 1007 |
| pointwise/norm/rope | 689 | 710 | 787 | 935 |
| other | 429 | 557 | 767 | 1131 |
| GPU busy total | 47.5 ms | 14.0 ms | 14.8 ms | 16.2 ms |

**End to end, the model GEMV wall dominates the diag_ell path** (10.8-11.0
ms/step, context-independent, ~83% of peak weight bandwidth): 63-73% of GPU
busy. Attention is second (10-14%), then a tail of pointwise/cache/other
(~15-20% combined, each item ~1 ms or less). In the *full* baseline the
ordering is different: at 128K its top cost is the DynamicCache append
(22.4 ms), then flash attention (13.4 ms), then GEMV (10.6 ms).

Consequences:

- Even making attention free would give diag_ell at most ~1.16x more e2e at
  128K. Larger e2e gains require attacking the GEMV wall (weight
  quantization ~1.4-1.55x, speculative decoding, batching) — orthogonal to
  the attention method.
- The attention method's value at batch 1 is therefore the attention-phase
  reduction itself (1.95-6.07x vs full), the KV-traffic reduction, and the
  robustness of decode latency to context length (14.75 -> 17.26 ms/step
  from 32K to 128K, vs 22.27 -> 49.61 for HF full attention).
