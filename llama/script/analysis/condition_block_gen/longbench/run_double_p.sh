#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-llama/result/generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p}"
P1="${P1:-0.95}"
P2="${P2:-0.70}"
CLUSTER_SIZE="${CLUSTER_SIZE:-32}"
KMEANS_ITERS="${KMEANS_ITERS:-4}"
SINK_TOKENS="${SINK_TOKENS:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"

read -r -a DATASET_LIST <<< "${DATASETS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" /usr/bin/time -f "WALL_TIME %E" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m llama.script.analysis.condition_block_gen.longbench.run_all \
    --device cuda:0 \
    --method double_p \
    --full-attention-layers 0 \
    --double-p-p1 "${P1}" \
    --double-p-p2 "${P2}" \
    --double-p-cluster-size "${CLUSTER_SIZE}" \
    --double-p-kmeans-iters "${KMEANS_ITERS}" \
    --double-p-sink-tokens "${SINK_TOKENS}" \
    --double-p-window-size "${WINDOW_SIZE}" \
    --datasets "${DATASET_LIST[@]}" \
    --output-root "${OUTPUT_ROOT}" \
    ${EXTRA_ARGS}
