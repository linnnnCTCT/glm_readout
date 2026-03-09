#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/outputs/hours_unlabeled_128k/dataset}"
HS_ROOT="${HS_ROOT:-$DATASET_ROOT/hidden_states}"
RUN_TAG="${RUN_TAG:-qformer_matrix_128k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/runs}"
EVAL_ROOT="${EVAL_ROOT:-$OUTPUT_ROOT/eval}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

TRAIN_GPU_GROUPS="${TRAIN_GPU_GROUPS:-0 1 2|3 4 5}"
EXPORT_GPUS_STRING="${EXPORT_GPUS:-0 1 2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29710}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"
MASK_RATIO="${MASK_RATIO:-0.30}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-0.01}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-2}"
SEQ_POOL="${SEQ_POOL:-attn}"

EXPERIMENTS=(
  "qformer_baseline|8|2|1e-5"
  "qformer_cov_5e-5|8|2|5e-5"
  "qformer_cov_1e-4|8|2|1e-4"
  "qformer_q16_cov_5e-5|16|2|5e-5"
)

TRAIN_LAUNCH_PID=""

if [[ ! -d "${HS_ROOT}/128k/train" ]]; then
  echo "Missing training hidden states: ${HS_ROOT}/128k/train" >&2
  exit 1
fi

for bucket in 8k 32k 128k; do
  if [[ ! -d "${HS_ROOT}/${bucket}/val" ]]; then
    echo "Missing validation hidden states: ${HS_ROOT}/${bucket}/val" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}"

IFS='|' read -r -a GPU_GROUPS <<< "${TRAIN_GPU_GROUPS}"
read -r -a EXPORT_GPUS <<< "${EXPORT_GPUS_STRING}"

if [[ "${#GPU_GROUPS[@]}" -eq 0 ]]; then
  echo "TRAIN_GPU_GROUPS must define at least one GPU group." >&2
  exit 1
fi
if [[ "${#EXPORT_GPUS[@]}" -lt 3 ]]; then
  echo "EXPORT_GPUS must define at least 3 GPU ids for 8k/32k/128k export." >&2
  exit 1
fi

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

launch_train_job() {
  local gpu_ids_string="$1"
  local master_port="$2"
  local log_file="$3"
  shift 3

  local world_size
  local visible_devices
  world_size="$(gpu_count "${gpu_ids_string}")"
  visible_devices="$(gpu_csv "${gpu_ids_string}")"

  if [[ "${world_size}" -le 0 ]]; then
    echo "No GPUs configured for training job." >&2
    return 1
  fi

  mkdir -p "$(dirname "${log_file}")"
  if [[ "${world_size}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
      python "$@" > "${log_file}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
      torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${world_size}" \
        --master_port="${master_port}" \
        "$@" > "${log_file}" 2>&1 &
  fi
  TRAIN_LAUNCH_PID="$!"
}

bucket_max_length() {
  case "$1" in
    8k) echo 8192 ;;
    32k) echo 32768 ;;
    128k) echo 131072 ;;
    *) echo "Unknown bucket $1" >&2; return 1 ;;
  esac
}

export_view() {
  local config_path="$1"
  local checkpoint_path="$2"
  local bucket="$3"
  local gpu="$4"
  local output_path="$5"
  local max_length
  max_length="$(bucket_max_length "${bucket}")"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${PROJECT_ROOT}/eval/export_embeddings.py" \
    --config "${config_path}" \
    --override \
      "data.data_root=${HS_ROOT}/${bucket}/val" \
      "data.hidden_dtype=bfloat16" \
      "data.max_length=${max_length}" \
      "data.random_crop=false" \
      "data.num_workers=${EXPORT_NUM_WORKERS}" \
    --checkpoint "${checkpoint_path}" \
    --output "${output_path}" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --num-workers "${EXPORT_NUM_WORKERS}"
}

echo "[1/3] Train matrix experiments"
WAVE=0
for ((start=0; start<${#EXPERIMENTS[@]}; start+=${#GPU_GROUPS[@]})); do
  PIDS=()
  ACTIVE_RUNS=()
  WAVE=$((WAVE + 1))
  echo "Launch wave ${WAVE}"

  for ((offset=0; offset<${#GPU_GROUPS[@]} && start+offset<${#EXPERIMENTS[@]}; offset++)); do
    EXP="${EXPERIMENTS[$((start + offset))]}"
    IFS='|' read -r RUN_NAME NUM_QUERIES NUM_LAYERS COV_WEIGHT <<< "${EXP}"

    RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
    CKPT_PATH="${RUN_DIR}/checkpoint_epoch_${TRAIN_EPOCHS}.pt"
    LOG_FILE="${LOG_ROOT}/train_${RUN_NAME}.log"
    GPU_GROUP="${GPU_GROUPS[$offset]}"
    MASTER_PORT=$((MASTER_PORT_BASE + start + offset))

    if [[ -f "${CKPT_PATH}" ]]; then
      echo "Skip training ${RUN_NAME}: checkpoint exists at ${CKPT_PATH}"
      continue
    fi

    echo "Launch training ${RUN_NAME} on GPUs [${GPU_GROUP}]"
    launch_train_job \
      "${GPU_GROUP}" \
      "${MASTER_PORT}" \
      "${LOG_FILE}" \
      "${PROJECT_ROOT}/train.py" \
      --config "${PROJECT_ROOT}/configs/v1.yaml" \
      --output-dir "${RUN_DIR}" \
      --override \
        "data.data_root=${HS_ROOT}/128k/train" \
        "data.hidden_dtype=bfloat16" \
        "data.max_length=131072" \
        "data.random_crop=false" \
        "data.num_workers=${TRAIN_NUM_WORKERS}" \
        "data.prefetch_factor=${TRAIN_PREFETCH_FACTOR}" \
        "training.batch_size=${TRAIN_BATCH_SIZE}" \
        "training.epochs=${TRAIN_EPOCHS}" \
        "training.ddp_find_unused_parameters=false" \
        "corruption.mask_ratio=${MASK_RATIO}" \
        "corruption.mask_target_span=true" \
        "loss.variance_weight=${VARIANCE_WEIGHT}" \
        "loss.covariance_weight=${COV_WEIGHT}" \
        "model.type=qformer" \
        "model.num_queries=${NUM_QUERIES}" \
        "model.num_layers=${NUM_LAYERS}" \
        "model.seq_pool=${SEQ_POOL}"
    PIDS+=("${TRAIN_LAUNCH_PID}")
    ACTIVE_RUNS+=("${RUN_NAME}")
  done

  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    wait "${PIDS[@]}"
  fi
  if [[ "${#ACTIVE_RUNS[@]}" -gt 0 ]]; then
    echo "Completed wave ${WAVE}: ${ACTIVE_RUNS[*]}"
  fi
done

echo "[2/3] Export embeddings and run retrieval"
RETRIEVAL_JSONS=()
for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r RUN_NAME NUM_QUERIES NUM_LAYERS COV_WEIGHT <<< "${EXP}"
  RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
  CONFIG_PATH="${RUN_DIR}/resolved_config.json"
  CKPT_PATH="${RUN_DIR}/checkpoint_epoch_${TRAIN_EPOCHS}.pt"
  EXP_EVAL_DIR="${EVAL_ROOT}/${RUN_NAME}"
  RETRIEVAL_PATH="${EXP_EVAL_DIR}/retrieval_consistency.json"
  mkdir -p "${EXP_EVAL_DIR}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Skip eval ${RUN_NAME}: missing checkpoint ${CKPT_PATH}" >&2
    continue
  fi

  EXPORT_PIDS=()
  for bucket_index in 0 1 2; do
    case "${bucket_index}" in
      0) BUCKET="8k" ;;
      1) BUCKET="32k" ;;
      2) BUCKET="128k" ;;
    esac
    EMB_PATH="${EXP_EVAL_DIR}/${BUCKET}_embeddings.pt"
    if [[ -f "${EMB_PATH}" ]]; then
      continue
    fi
    export_view "${CONFIG_PATH}" "${CKPT_PATH}" "${BUCKET}" "${EXPORT_GPUS[$bucket_index]}" "${EMB_PATH}" &
    EXPORT_PIDS+=("$!")
  done
  if [[ "${#EXPORT_PIDS[@]}" -gt 0 ]]; then
    wait "${EXPORT_PIDS[@]}"
  fi

  if [[ ! -f "${RETRIEVAL_PATH}" ]]; then
    python "${PROJECT_ROOT}/eval/retrieval_consistency.py" \
      --embeddings \
        8k="${EXP_EVAL_DIR}/8k_embeddings.pt" \
        32k="${EXP_EVAL_DIR}/32k_embeddings.pt" \
        128k="${EXP_EVAL_DIR}/128k_embeddings.pt" \
      --output "${RETRIEVAL_PATH}" \
      > "${LOG_ROOT}/retrieval_${RUN_NAME}.log" 2>&1
  fi
  RETRIEVAL_JSONS+=("${RETRIEVAL_PATH}")
done

echo "[3/3] Summarize matrix results"
if [[ "${#RETRIEVAL_JSONS[@]}" -gt 0 ]]; then
  python "${PROJECT_ROOT}/scripts/summarize_retrieval_consistency.py" \
    "${RETRIEVAL_JSONS[@]}" \
    --tsv-output "${OUTPUT_ROOT}/matrix_summary.tsv" \
    > "${OUTPUT_ROOT}/matrix_summary.txt"
fi

cat > "${OUTPUT_ROOT}/README_matrix_summary.txt" <<EOF
Run tag: ${RUN_TAG}
Dataset root: ${DATASET_ROOT}
Output root: ${OUTPUT_ROOT}
Training GPU groups: ${TRAIN_GPU_GROUPS}
Export GPUs: ${EXPORT_GPUS_STRING}

Experiments:
  qformer_baseline: num_queries=8, num_layers=2, covariance_weight=1e-5
  qformer_cov_5e-5: num_queries=8, num_layers=2, covariance_weight=5e-5
  qformer_cov_1e-4: num_queries=8, num_layers=2, covariance_weight=1e-4
  qformer_q16_cov_5e-5: num_queries=16, num_layers=2, covariance_weight=5e-5

Summary files:
  ${OUTPUT_ROOT}/matrix_summary.txt
  ${OUTPUT_ROOT}/matrix_summary.tsv
Logs:
  ${LOG_ROOT}
EOF

echo "Finished. Matrix summary: ${OUTPUT_ROOT}/matrix_summary.txt"
