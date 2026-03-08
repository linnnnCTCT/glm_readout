#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_launcher.sh"

HS_ROOT="${HS_ROOT:-$PROJECT_ROOT/outputs/feasibility_dataset/hidden_states}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/feasibility_runs}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/outputs/feasibility_eval}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_1M_PILOT="${RUN_1M_PILOT:-0}"
LAUNCHER="${LAUNCHER:-python}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
LOG_ROOT="${LOG_ROOT:-$EVAL_ROOT/logs/export_probe}"

mkdir -p "${EVAL_ROOT}"
mkdir -p "${LOG_ROOT}"
read -r -a GPU_IDS <<< "${GPU_IDS_STRING}"

JOBS=(
  "8k mean 10 8192"
  "8k attention 10 8192"
  "8k qformer 10 8192"
  "32k mean 10 32768"
  "32k qformer 10 32768"
  "128k qformer 10 131072"
)

if [[ "${RUN_1M_PILOT}" == "1" ]]; then
  JOBS+=("1m qformer 3 1048576")
fi

PIDS=()
JOB_INDEX=0
for JOB in "${JOBS[@]}"; do
  read -r BUCKET MODEL_TYPE EPOCH MAX_LENGTH <<< "${JOB}"
  CHECKPOINT_PATH="${RUN_ROOT}/${BUCKET}_${MODEL_TYPE}/checkpoint_epoch_${EPOCH}.pt"
  DATA_ROOT="${HS_ROOT}/${BUCKET}/test"
  OUTPUT_DIR="${EVAL_ROOT}/${BUCKET}_${MODEL_TYPE}"
  EMB_PATH="${OUTPUT_DIR}/embeddings.pt"
  JSON_PATH="${OUTPUT_DIR}/linear_probe.json"
  GPU="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  MASTER_PORT=$((MASTER_PORT_BASE + JOB_INDEX))
  LOG_FILE="${LOG_ROOT}/${BUCKET}_${MODEL_TYPE}.log"

  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Skip missing checkpoint: ${CHECKPOINT_PATH}"
    JOB_INDEX=$((JOB_INDEX + 1))
    continue
  fi

  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Skip missing eval data directory: ${DATA_ROOT}"
    JOB_INDEX=$((JOB_INDEX + 1))
    continue
  fi

  mkdir -p "${OUTPUT_DIR}"

  echo "Export embeddings: bucket=${BUCKET} model=${MODEL_TYPE} gpu=${GPU} launcher=${LAUNCHER}"
  launch_single_gpu_job \
    "${LAUNCHER}" \
    "${GPU}" \
    "${LOG_FILE}" \
    "${MASTER_PORT}" \
    bash -lc "
      set -euo pipefail
      python '${PROJECT_ROOT}/eval/export_embeddings.py' \
        --config '${PROJECT_ROOT}/configs/v1.yaml' \
        --override \
          'data.data_root=${DATA_ROOT}' \
          'data.hidden_dtype=bfloat16' \
          'data.max_length=${MAX_LENGTH}' \
          'data.random_crop=false' \
        --checkpoint '${CHECKPOINT_PATH}' \
        --output '${EMB_PATH}' \
        --batch-size 4

      if python -c \"import sys, torch; payload=torch.load(sys.argv[1], map_location='cpu'); raise SystemExit(0 if 'labels' in payload else 1)\" '${EMB_PATH}'; then
        python '${PROJECT_ROOT}/eval/linear_probe.py' \
          --embeddings '${EMB_PATH}' \
          --epochs 50 \
          --lr 1e-3 \
          --task auto \
          --output '${JSON_PATH}'
      else
        echo 'No labels found in ${EMB_PATH}; skip linear probe.'
      fi
    "
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

echo "Export and probe finished: ${EVAL_ROOT}"
