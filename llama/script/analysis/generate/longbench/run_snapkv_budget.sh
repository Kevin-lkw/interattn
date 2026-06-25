#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-5}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-llama/result/generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique }"
KVPRESS_WINDOW_SIZE="${KVPRESS_WINDOW_SIZE:-64}"
KVPRESS_KERNEL_SIZE="${KVPRESS_KERNEL_SIZE:-5}"

METHOD="kvpress_snapkv"

ENV_PREFIX=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
)

if [ "$#" -gt 0 ]; then
  BUDGET_LIST=("$@")
else
  BUDGET_LIST=(0.1 0.25 0.5 0.75)
fi

read -r -a DATASET_LIST <<< "${DATASETS}"

for DATASET in "${DATASET_LIST[@]}"; do
  echo "=== ${METHOD} dataset=${DATASET}, budget_count=${#BUDGET_LIST[@]} ==="
  for BUDGET in "${BUDGET_LIST[@]}"; do
    RUN_OUTPUT_ROOT="${OUTPUT_ROOT}"
    echo "=== ${METHOD} dataset=${DATASET}, budget=${BUDGET}, gpu=${GPU_ID}, output=${RUN_OUTPUT_ROOT} ==="
    env "${ENV_PREFIX[@]}" /usr/bin/time -f "WALL_TIME %E" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m llama.script.analysis.generate.longbench.run_all \
        --method "${METHOD}" \
        --budget "${BUDGET}" \
        --kvpress-window-size "${KVPRESS_WINDOW_SIZE}" \
        --kvpress-kernel-size "${KVPRESS_KERNEL_SIZE}" \
        --datasets "${DATASET}" \
        --output-root "${RUN_OUTPUT_ROOT}" \
        ${EXTRA_ARGS}
  done
done
