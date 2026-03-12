#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/qformer_q16_cov_5e-5_curriculum.yaml}"

HS_ROOT="${HS_ROOT:-$PROJECT_ROOT/outputs/feasibility_dataset/hidden_states}"
RUN_TAG="${RUN_TAG:-qformer_q16_cov_5e-5_curriculum}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/runs}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29810}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-contextagg-readout}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

PHASE_SPECS=(
  "p1_bootstrap|32768|6|1e-4|256|8k,32k|8k=0.70,32k=0.30|8k=16,32k=8"
  "p2_bridge|131072|5|7e-5|256|8k,32k,128k|8k=0.20,32k=0.60,128k=0.20|8k=12,32k=8,128k=4"
  "p3_long|1048576|4|4e-5|128|32k,128k,1m|32k=0.20,128k=0.70,1m=0.10|32k=8,128k=4,1m=1"
  "p4_adapt|1048576|2|2e-5|128|128k,1m|128k=0.80,1m=0.20|128k=4,1m=1"
)

gpu_count() {
  local gpu_ids_string="$1"
  local -a gpu_ids=()
  read -r -a gpu_ids <<< "${gpu_ids_string}"
  echo "${#gpu_ids[@]}"
}

gpu_csv() {
  local gpu_ids_string="$1"
  local -a gpu_ids=()
  read -r -a gpu_ids <<< "${gpu_ids_string}"
  local joined=""
  local gpu
  for gpu in "${gpu_ids[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${gpu}"
  done
  echo "${joined}"
}

csv_to_python_list() {
  local csv="$1"
  local result="["
  local first=1
  local item
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${first}" -eq 0 ]]; then
      result+=", "
    fi
    result+="\"${item}\""
    first=0
  done
  result+="]"
  echo "${result}"
}

kv_csv_to_python_dict() {
  local csv="$1"
  local result="{"
  local first=1
  local pair key value
  IFS=',' read -r -a pairs <<< "${csv}"
  for pair in "${pairs[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    if [[ "${first}" -eq 0 ]]; then
      result+=", "
    fi
    result+="\"${key}\": ${value}"
    first=0
  done
  result+="}"
  echo "${result}"
}

bucket_roots_to_python_list() {
  local csv="$1"
  local result="["
  local first=1
  local bucket root
  IFS=',' read -r -a buckets <<< "${csv}"
  for bucket in "${buckets[@]}"; do
    root="${HS_ROOT}/${bucket}/train"
    if [[ ! -d "${root}" ]]; then
      echo "Missing hidden-state directory: ${root}" >&2
      exit 1
    fi
    if [[ "${first}" -eq 0 ]]; then
      result+=", "
    fi
    result+="\"${root}\""
    first=0
  done
  result+="]"
  echo "${result}"
}

launch_train_job() {
  local gpu_ids_string="$1"
  local master_port="$2"
  local log_file="$3"
  shift 3

  local world_size
  local visible_devices
  world_size="$(gpu_count "${gpu_ids_string}")"
  visible_devices="$(gpu_csv "${gpu_ids_string}")"
  mkdir -p "$(dirname "${log_file}")"

  if [[ "${world_size}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${visible_devices}" python "$@" > "${log_file}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${visible_devices}" torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${world_size}" \
      --master_port="${master_port}" \
      "$@" > "${log_file}" 2>&1 &
  fi
  LAUNCH_PID="$!"
}

mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

PREV_CHECKPOINT=""
PHASE_INDEX=0
for PHASE in "${PHASE_SPECS[@]}"; do
  IFS='|' read -r PHASE_NAME MAX_LENGTH EPOCHS LR STEPS_PER_EPOCH BUCKETS BUCKET_WEIGHTS BUCKET_BATCH_SIZES <<< "${PHASE}"
  PHASE_DIR="${RUN_ROOT}/${PHASE_NAME}"
  LOG_FILE="${LOG_ROOT}/${PHASE_NAME}.log"
  CHECKPOINT_PATH="${PHASE_DIR}/checkpoint_epoch_${EPOCHS}.pt"
  MASTER_PORT=$((MASTER_PORT_BASE + PHASE_INDEX))

  BUCKET_ROOTS="$(bucket_roots_to_python_list "${BUCKETS}")"
  BUCKET_NAMES="$(csv_to_python_list "${BUCKETS}")"
  BUCKET_WEIGHTS_DICT="$(kv_csv_to_python_dict "${BUCKET_WEIGHTS}")"
  BUCKET_BATCH_SIZES_DICT="$(kv_csv_to_python_dict "${BUCKET_BATCH_SIZES}")"

  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    echo "Skip ${PHASE_NAME}: found ${CHECKPOINT_PATH}"
    PREV_CHECKPOINT="${CHECKPOINT_PATH}"
    PHASE_INDEX=$((PHASE_INDEX + 1))
    continue
  fi

  CMD=(
    "${PROJECT_ROOT}/train.py"
    --config "${CONFIG_PATH}"
    --output-dir "${PHASE_DIR}"
    --override
    "data.data_roots=${BUCKET_ROOTS}"
    "data.data_root_names=${BUCKET_NAMES}"
    "data.bucket_weights=${BUCKET_WEIGHTS_DICT}"
    "data.bucket_batch_sizes=${BUCKET_BATCH_SIZES_DICT}"
    "data.steps_per_epoch=${STEPS_PER_EPOCH}"
    "data.max_length=${MAX_LENGTH}"
    "data.random_crop=false"
    "data.hidden_dtype=bfloat16"
    "data.num_workers=${TRAIN_NUM_WORKERS}"
    "data.prefetch_factor=${TRAIN_PREFETCH_FACTOR}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.batch_size=12"
    "training.ddp_find_unused_parameters=false"
    "wandb.enabled=$( [[ "${WANDB_ENABLED}" == "1" ]] && echo True || echo False )"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${RUN_TAG}_${PHASE_NAME}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=("wandb.entity=${WANDB_ENTITY}")
  fi
  if [[ -n "${PREV_CHECKPOINT}" ]]; then
    CMD+=(--init-checkpoint "${PREV_CHECKPOINT}")
  fi

  echo "Launch ${PHASE_NAME}: buckets=${BUCKETS} lr=${LR} epochs=${EPOCHS} gpus=[${TRAIN_GPUS}]"
  launch_train_job "${TRAIN_GPUS}" "${MASTER_PORT}" "${LOG_FILE}" "${CMD[@]}"
  wait "${LAUNCH_PID}"

  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Phase ${PHASE_NAME} did not produce expected checkpoint: ${CHECKPOINT_PATH}" >&2
    exit 1
  fi

  PREV_CHECKPOINT="${CHECKPOINT_PATH}"
  PHASE_INDEX=$((PHASE_INDEX + 1))
done

cat > "${OUTPUT_ROOT}/README_curriculum.txt" <<EOF
Config: ${CONFIG_PATH}
Hidden-state root: ${HS_ROOT}
Run root: ${RUN_ROOT}
Logs: ${LOG_ROOT}
Train GPUs: ${TRAIN_GPUS}
Wandb enabled: ${WANDB_ENABLED}
Wandb project: ${WANDB_PROJECT}
Wandb entity: ${WANDB_ENTITY}
Wandb mode: ${WANDB_MODE}

Phases:
  p1_bootstrap: 8k/32k, weights 0.70/0.30, batch sizes 12/6, 256 steps/epoch, 6 epochs, lr 1e-4
  p2_bridge: 8k/32k/128k, weights 0.20/0.60/0.20, batch sizes 10/6/3, 256 steps/epoch, 5 epochs, lr 7e-5
  p3_long: 32k/128k/1m, weights 0.20/0.70/0.10, batch sizes 6/3/1, 128 steps/epoch, 4 epochs, lr 4e-5
  p4_adapt: 128k/1m, weights 0.80/0.20, batch sizes 2/1, 128 steps/epoch, 2 epochs, lr 2e-5
EOF

echo "Curriculum training finished. Final checkpoint: ${PREV_CHECKPOINT}"
