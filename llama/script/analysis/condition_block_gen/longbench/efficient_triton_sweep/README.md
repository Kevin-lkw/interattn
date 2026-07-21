# Efficient Triton full LongBench sweep

This experiment compares `block_size={32,64}` and
`eps={0.05,0.1,0.25,0.5}` on the existing 16-task English/code LongBench
suite (3,750 samples per configuration). The five Chinese tasks are excluded
from this first sweep. The runtime configuration shared by every point is:

```text
fused Triton selection + page attention
mixed summaries + BF16 k_bar
post-prefill StaticCache
stats disabled
TMA disabled
CUDA graph disabled
```

CUDA graph is intentionally disabled.  Full LongBench contains many
early-EOS tasks; graph replay executes the configured maximum generation
length before truncation and is therefore slower for this workload.  Each
JSONL row records its actual end-to-end `generation_seconds` with stats off.

Run or resume on at most four exclusive GPUs in tmux:

```bash
./llama/script/analysis/condition_block_gen/longbench/efficient_triton_sweep/launch_tmux.sh \
  0 2 4 5
```

The scheduler assigns one complete `(block_size, eps)` configuration to a GPU,
loads the model once per configuration, and starts the next queued job when a
GPU finishes. Existing IDs are skipped by the standard LongBench runner. The
tmux entry point runs the summarizer automatically after all jobs finish.

Summarize accuracy, completeness and observed latency:

```bash
python -m llama.script.analysis.condition_block_gen.longbench.efficient_triton_sweep.summarize \
  --result-root llama/result/generate/condition_block_efficient_full_sweep
```

Generated artifacts:

- `sweep_manifest.json`: job status and GPU assignment;
- `sweep_logs/*.log`: per-configuration logs;
- `sweep_summary/summary.json`: machine-readable config and dataset results;
- `sweep_summary/datasets.csv`: per-dataset table;
- `sweep_summary/RESULTS.md`: human-readable score and latency tables.

## Results

All eight configurations completed 16/16 datasets and 3,750/3,750 samples.
Latency is the sample-weighted end-to-end generation time, including prefill
and decode. Statistics were disabled on every run.

| block | eps | macro score | mean latency/sample | output tok/s |
|---:|---:|---:|---:|---:|
| 32 | 0.05 | 49.12 | 2.079 s | 41.10 |
| 32 | 0.10 | 48.85 | 2.058 s | 41.40 |
| 32 | 0.25 | 47.98 | 1.997 s | 42.27 |
| 32 | 0.50 | 46.57 | 1.981 s | 43.05 |
| 64 | 0.05 | **49.16** | 2.118 s | 40.30 |
| 64 | 0.10 | 49.02 | 2.084 s | 40.63 |
| 64 | 0.25 | 48.30 | 2.028 s | 41.65 |
| 64 | 0.50 | 47.35 | 1.987 s | 42.52 |

At matched epsilon, block64 retained `+0.04/+0.17/+0.32/+0.78` macro score,
but its end-to-end latency was `0.982x/0.988x/0.985x/0.997x` block32 speed
(a value below 1 means block64 was slower). On this mixed suite, block64's
attention-stage advantage is diluted by prefill, short generations, model
GEMV and its looser routing bound. Block32 is the better latency choice;
block64 is useful only when its modest quality gain matters more.

Detailed per-dataset score, mean/median/p95 latency and throughput are in
`llama/result/generate/condition_block_efficient_full_sweep/sweep_summary/RESULTS.md`.
