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

Final results will be copied into this README after all eight configurations
reach 16/16 complete datasets.
