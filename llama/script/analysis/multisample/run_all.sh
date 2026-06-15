#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GPU_ID="${GPU_ID:-7}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEQ_LEN="${SEQ_LEN:-1024}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-${SEQ_LEN}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LLAMA_DIR}/result/wikitext_n100}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${LLAMA_DIR}"

run_python() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python "$@"
}

run_method() {
  local module="$1"
  local log_name="$2"
  shift 2
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting ${module}"
  run_python -m "${module}" \
    --device cuda:0 \
    --num-samples "${NUM_SAMPLES}" \
    --seq-len "${SEQ_LEN}" \
    --sample-stride "${SAMPLE_STRIDE}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished ${module}"
}

echo "GPU_ID=${GPU_ID}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "SAMPLE_STRIDE=${SAMPLE_STRIDE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

run_python -m script.analysis.multisample.verify_dataset \
  --num-samples "${NUM_SAMPLES}" \
  --seq-len "${SEQ_LEN}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  2>&1 | tee "${LOG_DIR}/verify_dataset.log"

run_method \
  script.analysis.multisample.run_attention_topk \
  attention_topk

run_method \
  script.analysis.multisample.run_condition_block \
  condition_block

run_method \
  script.analysis.multisample.run_h2o \
  h2o

run_method \
  script.analysis.multisample.run_quest \
  quest

run_python -m script.analysis.multisample.plot_all \
  --output-root "${OUTPUT_ROOT}" \
  --require-samples "${NUM_SAMPLES}" \
  2>&1 | tee "${LOG_DIR}/plot_all.log"

echo "All experiments completed."
echo "Combined plot: ${OUTPUT_ROOT}/all_methods_mean_ppl_vs_budget.png"
