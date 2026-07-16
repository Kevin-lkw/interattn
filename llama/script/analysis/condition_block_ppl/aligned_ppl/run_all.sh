#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

GPU_ID="${GPU_ID:-7}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
MODEL_NAME="${MODEL_NAME:-${MODEL##*/}}"
DTYPE="${DTYPE:-bfloat16}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-${SEQ_LEN}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LLAMA_DIR}/result/${MODEL_NAME}/aligned_ppl}"
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
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --ppl-protocol aligned \
    --num-samples "${NUM_SAMPLES}" \
    --seq-len "${SEQ_LEN}" \
    --sample-stride "${SAMPLE_STRIDE}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Finished ${module}"
}

echo "GPU_ID=${GPU_ID}"
echo "MODEL=${MODEL}"
echo "DTYPE=${DTYPE}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "SAMPLE_STRIDE=${SAMPLE_STRIDE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

run_python -m script.analysis.condition_block_ppl.aligned_ppl.sanity_check \
  --model "${MODEL}" --dtype "${DTYPE}" --device cuda:0 \
  --seq-len "${SEQ_LEN}" \
  --output "${OUTPUT_ROOT}/sanity/sanity.json" \
  2>&1 | tee "${LOG_DIR}/sanity.log"

run_python -m script.analysis.condition_block_ppl.multisample.verify_dataset \
  --model "${MODEL}" --ppl-protocol aligned \
  --num-samples "${NUM_SAMPLES}" --seq-len "${SEQ_LEN}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  2>&1 | tee "${LOG_DIR}/verify_dataset.log"

run_method script.analysis.condition_block_ppl.multisample.run_attention_topk attention_topk
run_method script.analysis.condition_block_ppl.multisample.run_condition_block condition_block
run_method script.analysis.condition_block_ppl.multisample.run_double_p double_p
run_method script.analysis.condition_block_ppl.multisample.run_h2o h2o
run_method script.analysis.condition_block_ppl.multisample.run_streamllm streamllm
run_method script.analysis.condition_block_ppl.multisample.run_quest quest

run_python -m script.analysis.condition_block_ppl.multisample.plot_all \
  --output-root "${OUTPUT_ROOT}" \
  --require-samples "${NUM_SAMPLES}" \
  --methods condition_block double_p attention_topk h2o streamllm quest \
  --metric corpus_ppl --xscale log --include-all-settings \
  2>&1 | tee "${LOG_DIR}/plot_all.log"

echo "All aligned PPL experiments completed."
echo "Combined plot: ${OUTPUT_ROOT}/all_methods_corpus_ppl_vs_budget.png"
