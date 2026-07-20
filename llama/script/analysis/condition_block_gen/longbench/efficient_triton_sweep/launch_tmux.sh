#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 4 )); then
  echo "usage: $0 GPU_ID [GPU_ID ...]" >&2
  exit 2
fi

session="condition_block_full_sweep"
repo="/scratch1/liankewei/interattn"
python="/scratch1/liankewei/miniconda3/envs/nanogpt/bin/python"
output="$repo/llama/result/generate/condition_block_efficient_full_sweep"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

gpu_args=""
for gpu in "$@"; do
  gpu_args+=" $gpu"
done

command="cd '$repo' && PYTHONPATH='$repo' '$python' -m llama.script.analysis.condition_block_gen.longbench.efficient_triton_sweep.run --gpus$gpu_args --output-root '$output' 2>&1 | tee '$output/tmux_driver.log'"
mkdir -p "$output"
tmux new-session -d -s "$session" -n sweep "$command"

echo "started tmux session: $session"
echo "attach: tmux attach -t $session"
echo "driver log: $output/tmux_driver.log"
