#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_launcher.sh"

HS_ROOT="${HS_ROOT:-$PROJECT_ROOT/outputs/feasibility_dataset/hidden_states}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/feasibility_runs}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_1M_PILOT="${RUN_1M_PILOT:-0}"
LAUNCHER="${LAUNCHER:-python}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs/train}"

if [[ ! -d "${HS_ROOT}" ]]; then
  echo "HS_ROOT does not exist: ${HS_ROOT}" >&2
  exit 1
fi

read -r -a GPU_IDS <<< "${GPU_IDS_STRING}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${LOG_ROOT}"

JOBS=(
  "8k mean 8192 8 10 64 64 16"
  "8k attention 8192 8 10 64 64 16"
  "8k qformer 8192 8 10 64 64 32"
  "32k mean 32768 4 10 64 64 16"
  "32k qformer 32768 4 10 64 64 32"
  "128k qformer 131072 2 10 64 64 32"
)

if [[ "${RUN_1M_PILOT}" == "1" ]]; then
  JOBS+=("1m qformer 1048576 1 3 64 64 32")
fi

PIDS=()
JOB_INDEX=0
for JOB in "${JOBS[@]}"; do
  read -r BUCKET MODEL_TYPE MAX_LENGTH BATCH_SIZE EPOCHS CHUNK_SIZE STRIDE NUM_QUERIES <<< "${JOB}"
  DATA_ROOT="${HS_ROOT}/${BUCKET}/train"
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Skip missing hidden-state directory: ${DATA_ROOT}"
    continue
  fi

  OUTPUT_DIR="${RUN_ROOT}/${BUCKET}_${MODEL_TYPE}"
  GPU="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  MASTER_PORT=$((MASTER_PORT_BASE + JOB_INDEX))
  LOG_FILE="${LOG_ROOT}/${BUCKET}_${MODEL_TYPE}.log"

  echo "Launch training: bucket=${BUCKET} model=${MODEL_TYPE} gpu=${GPU} launcher=${LAUNCHER}"
  launch_single_gpu_job \
    "${LAUNCHER}" \
    "${GPU}" \
    "${LOG_FILE}" \
    "${MASTER_PORT}" \
    python "${PROJECT_ROOT}/train.py" \
      --config "${PROJECT_ROOT}/configs/v1.yaml" \
      --output-dir "${OUTPUT_DIR}" \
      --override \
        "data.data_root=${DATA_ROOT}" \
        "data.hidden_dtype=bfloat16" \
        "data.max_length=${MAX_LENGTH}" \
        "data.random_crop=false" \
        "training.batch_size=${BATCH_SIZE}" \
        "training.epochs=${EPOCHS}" \
        "model.type=${MODEL_TYPE}" \
        "model.num_queries=${NUM_QUERIES}" \
        "model.chunk_pooling.chunk_size=${CHUNK_SIZE}" \
        "model.chunk_pooling.stride=${STRIDE}"
  PIDS+=("${LAUNCH_PID}")
  JOB_INDEX=$((JOB_INDEX + 1))

  if (( JOB_INDEX % ${#GPU_IDS[@]} == 0 )); then
    wait_for_pids "${PIDS[@]}"
    PIDS=()
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_for_pids "${PIDS[@]}"
fi

echo "Training finished: ${RUN_ROOT}"
