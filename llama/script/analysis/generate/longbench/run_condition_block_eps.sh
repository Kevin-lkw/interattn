#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result/condition_block_eps_batched}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ENV_PREFIX=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
)

if [ "$#" -gt 0 ]; then
  EPS_LIST=("$@")
else
  EPS_LIST=(0.05 0.1 0.25 0.5)
fi

for EPS in "${EPS_LIST[@]}"; do
  EPS_TAG="${EPS//./p}"
  RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/eps_${EPS_TAG}"
  echo "=== condition_block eps=${EPS}, block_size=${BLOCK_SIZE}, gpu=${GPU_ID}, output=${RUN_OUTPUT_ROOT} ==="
  env "${ENV_PREFIX[@]}" /usr/bin/time -f "WALL_TIME %E" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m llama.script.analysis.generate.longbench.run_all \
      --method condition_block \
      --condition-block-size "${BLOCK_SIZE}" \
      --condition-eps "${EPS}" \
      --output-root "${RUN_OUTPUT_ROOT}" \
      ${EXTRA_ARGS}
done
