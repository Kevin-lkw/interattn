# Multi-sample evaluation

The four runners evaluate 100 non-overlapping WikiText-2 test windows by
default. Window starts are:

```text
0, 1024, 2048, ..., 101376
```

Each process loads the model once, saves after every completed sample, and
resumes by skipping samples already present in its `summary.pt`.

WikiText-2 test contains 335,645 Llama-2 tokens. The default 100-window run
requires 102,401 tokens, so the dataset has sufficient capacity.

To run all four methods sequentially and produce one combined plot:

```bash
GPU_ID=7 bash script/analysis/multisample/run_all.sh
```

This command is suitable for a tmux session. Per-method logs are written to
`result/wikitext_n100/logs/`. Re-running the command resumes completed methods
from their per-sample checkpoints.

Run from the `llama` directory:

```bash
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.multisample.run_attention_topk --device cuda:0

CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.multisample.run_condition_block --device cuda:0

CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.multisample.run_h2o --device cuda:0

CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n nanogpt \
  python -m script.analysis.multisample.run_quest --device cuda:0
```

Default outputs:

```text
result/wikitext_n100/<method>/summary.pt
result/wikitext_n100/<method>/mean_ppl_vs_budget.png
result/wikitext_n100/all_methods_mean_ppl_vs_budget.png
```

`summary.pt` contains every sample's NLL/PPL and aggregate statistics:

- `mean_ppl`: arithmetic mean of the 100 per-sample PPL values.
- `std_ppl`: population standard deviation of per-sample PPL.
- `mean_nll`: arithmetic mean of per-sample NLL.
- `corpus_ppl`: `exp(mean_nll)`.
- `mean_measured_budget`: mean equivalent causal attention budget.

Method defaults:

- Attention top-k: first 2 layers use full attention.
- Condition-block: first 2 layers use full attention, block size 10.
- H2O: all layers are compressed.
- QUEST: first 2 layers use full attention, page size 16.
