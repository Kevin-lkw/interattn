#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-4}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-llama/result/generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p}"

ENV_PREFIX=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
)

if [ "$#" -gt 0 ]; then
  EPS_LIST=("$@")
else
  EPS_LIST=(0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5)
fi

read -r -a DATASET_LIST <<< "${DATASETS}"

for DATASET in "${DATASET_LIST[@]}"; do
  echo "=== condition_block dataset=${DATASET}, eps_count=${#EPS_LIST[@]} ==="
  for EPS in "${EPS_LIST[@]}"; do
    RUN_OUTPUT_ROOT="${OUTPUT_ROOT}"
    echo "=== condition_block dataset=${DATASET}, eps=${EPS}, block_size=${BLOCK_SIZE}, gpu=${GPU_ID}, output=${RUN_OUTPUT_ROOT} ==="
    env "${ENV_PREFIX[@]}" /usr/bin/time -f "WALL_TIME %E" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m llama.script.analysis.condition_block_gen.longbench.run_all \
        --method condition_block \
        --condition-block-size "${BLOCK_SIZE}" \
        --condition-eps "${EPS}" \
        --datasets "${DATASET}" \
        --output-root "${RUN_OUTPUT_ROOT}" \
        ${EXTRA_ARGS}
  done
done
