#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-llama/result/generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
COLLECT_STATS="${COLLECT_STATS:-1}"
FAST_SKIP="${FAST_SKIP:-1}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p}"
TASK_INFO="${TASK_INFO:-llama/script/analysis/condition_block_gen/longbench/task_info.json}"

ENV_PREFIX=(
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
)
if [ "${COLLECT_STATS}" = "0" ]; then
  ENV_PREFIX+=("CONDITION_BLOCK_SKIP_STATS=1")
fi

if [ "$#" -gt 0 ]; then
  EPS_LIST=("$@")
else
  EPS_LIST=(0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5)
fi

read -r -a DATASET_LIST <<< "${DATASETS}"

fast_skip_complete() {
  local dataset="$1"
  local eps="$2"
  local out_path="${OUTPUT_ROOT}/${MODEL_NAME}/longbench/${dataset}/condition_block_triton_block=${BLOCK_SIZE}_eps=${eps}.jsonl"

  [ "${FAST_SKIP}" = "1" ] || return 1
  [ -f "${out_path}" ] || return 1
  [ -f "${TASK_INFO}" ] || return 1

  python - "${TASK_INFO}" "${dataset}" "${out_path}" "${eps}" <<'PY'
import json
import sys

task_info_path, dataset, out_path, eps = sys.argv[1:5]
with open(task_info_path, "r", encoding="utf-8") as handle:
    expected = int(json.load(handle)[dataset]["num_test"])

done = set()
with open(out_path, "r", encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in row:
            done.add(str(row["id"]))

if len(done) >= expected:
    print(f"=== fast-skip complete {dataset} eps={eps} ({len(done)}/{expected} ids): {out_path} ===")
    raise SystemExit(0)
raise SystemExit(1)
PY
}

for DATASET in "${DATASET_LIST[@]}"; do
  echo "=== condition_block_triton dataset=${DATASET}, eps_count=${#EPS_LIST[@]} ==="
  for EPS in "${EPS_LIST[@]}"; do
    RUN_OUTPUT_ROOT="${OUTPUT_ROOT}"
    echo "=== condition_block_triton dataset=${DATASET}, eps=${EPS}, block_size=${BLOCK_SIZE}, gpu=${GPU_ID}, collect_stats=${COLLECT_STATS}, output=${RUN_OUTPUT_ROOT} ==="
    if fast_skip_complete "${DATASET}" "${EPS}"; then
      continue
    fi
    env "${ENV_PREFIX[@]}" /usr/bin/time -f "WALL_TIME %E" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      python -m llama.script.analysis.condition_block_gen.longbench.run_all \
        --method condition_block_triton \
        --condition-block-size "${BLOCK_SIZE}" \
        --condition-eps "${EPS}" \
        --datasets "${DATASET}" \
        --output-root "${RUN_OUTPUT_ROOT}" \
        ${EXTRA_ARGS}
  done
done
