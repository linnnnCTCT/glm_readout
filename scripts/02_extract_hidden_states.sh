#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_launcher.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-2-5B-8k}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/outputs/feasibility_dataset}"
HS_ROOT="${HS_ROOT:-$DATASET_ROOT/hidden_states}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LAUNCHER="${LAUNCHER:-python}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
LOG_ROOT="${LOG_ROOT:-$DATASET_ROOT/logs/extract}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}/manifests" ]]; then
  echo "Manifest directory not found: ${DATASET_ROOT}/manifests" >&2
  exit 1
fi

read -r -a GPU_IDS <<< "${GPU_IDS_STRING}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU index." >&2
  exit 1
fi

mkdir -p "${HS_ROOT}"
mkdir -p "${LOG_ROOT}"

manifest_count() {
  local manifest_path="$1"
  wc -l < "${manifest_path}"
}

hidden_count() {
  local output_dir="$1"
  if [[ ! -d "${output_dir}" ]]; then
    echo 0
    return 0
  fi
  find "${output_dir}" -maxdepth 1 -type f -name '*.pt' | wc -l
}

JOBS=(
  "8k train 8192 4"
  "8k val 8192 4"
  "8k test 8192 4"
  "32k train 32768 2"
  "32k val 32768 2"
  "32k test 32768 2"
  "128k train 131072 1"
  "128k val 131072 1"
  "128k test 131072 1"
  "1m train 1048576 1"
  "1m val 1048576 1"
  "1m test 1048576 1"
)

PIDS=()
JOB_INDEX=0
for JOB in "${JOBS[@]}"; do
  read -r BUCKET SPLIT MAX_LENGTH BATCH_SIZE <<< "${JOB}"
  MANIFEST_PATH="${DATASET_ROOT}/manifests/${BUCKET}/${SPLIT}.jsonl"
  OUTPUT_DIR="${HS_ROOT}/${BUCKET}/${SPLIT}"

  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Skip missing manifest: ${MANIFEST_PATH}"
    continue
  fi

  MANIFEST_SAMPLES="$(manifest_count "${MANIFEST_PATH}")"
  if [[ "${MANIFEST_SAMPLES}" -eq 0 ]]; then
    echo "Skip empty manifest: ${MANIFEST_PATH}"
    continue
  fi

  EXISTING_SAMPLES="$(hidden_count "${OUTPUT_DIR}")"
  if [[ "${EXISTING_SAMPLES}" -eq "${MANIFEST_SAMPLES}" ]]; then
    echo "Skip completed split: bucket=${BUCKET} split=${SPLIT} samples=${MANIFEST_SAMPLES}"
    continue
  fi

  GPU="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  MASTER_PORT=$((MASTER_PORT_BASE + JOB_INDEX))
  LOG_FILE="${LOG_ROOT}/${BUCKET}_${SPLIT}.log"
  mkdir -p "${OUTPUT_DIR}"

  echo "Launch extraction: bucket=${BUCKET} split=${SPLIT} gpu=${GPU} launcher=${LAUNCHER}"
  launch_single_gpu_job \
    "${LAUNCHER}" \
    "${GPU}" \
    "${LOG_FILE}" \
    "${MASTER_PORT}" \
    python "${PROJECT_ROOT}/data/extract_hidden_states.py" \
      --model-path "${MODEL_PATH}" \
      --input "${MANIFEST_PATH}" \
      --input-format jsonl \
      --output-dir "${OUTPUT_DIR}" \
      --max-length "${MAX_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --device cuda \
      --model-dtype "${MODEL_DTYPE}" \
      --save-dtype "${SAVE_DTYPE}" \
      --skip-existing
  PIDS+=("${LAUNCH_PID}")
  JOB_INDEX=$((JOB_INDEX + 1))

  if (( ${#PIDS[@]} == ${#GPU_IDS[@]} )); then
    wait_for_pids "${PIDS[@]}"
    PIDS=()
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_for_pids "${PIDS[@]}"
fi

echo "Hidden-state extraction finished: ${HS_ROOT}"
