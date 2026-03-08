#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FASTA_LIST="${FASTA_LIST:-$PROJECT_ROOT/MAGsANI97.txt}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-2-5B-8k}"
RUN_TAG="${RUN_TAG:-hours_unlabeled_128k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/dataset}"
HS_ROOT="${HS_ROOT:-$DATASET_ROOT/hidden_states}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/runs}"
EVAL_ROOT="${EVAL_ROOT:-$OUTPUT_ROOT/eval}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

WORKERS="${WORKERS:-10}"
SEED="${SEED:-42}"

TRAIN_GENOMES_128K="${TRAIN_GENOMES_128K:-192}"
VAL_GENOMES_SHARED="${VAL_GENOMES_SHARED:-64}"
WINDOWS_128K_TRAIN="${WINDOWS_128K_TRAIN:-2}"
WINDOWS_128K_VAL="${WINDOWS_128K_VAL:-1}"

EXTRACT_GPU_IDS="${EXTRACT_GPU_IDS:-0 1 2 3 4 5 6}"
TRAIN_GPU_IDS_QFORMER="${TRAIN_GPU_IDS_QFORMER:-0 1 2 3}"
TRAIN_GPU_IDS_MEAN="${TRAIN_GPU_IDS_MEAN:-4 5 6}"
EXPORT_GPUS_STRING="${EXPORT_GPUS:-0 1 2}"
TRAIN_MASTER_PORT_QFORMER="${TRAIN_MASTER_PORT_QFORMER:-29610}"
TRAIN_MASTER_PORT_MEAN="${TRAIN_MASTER_PORT_MEAN:-29620}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-2}"
MASK_RATIO="${MASK_RATIO:-0.30}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-0.01}"
COVARIANCE_WEIGHT="${COVARIANCE_WEIGHT:-1e-5}"
QFORMER_NUM_QUERIES="${QFORMER_NUM_QUERIES:-8}"
QFORMER_NUM_LAYERS="${QFORMER_NUM_LAYERS:-2}"
QFORMER_SEQ_POOL="${QFORMER_SEQ_POOL:-attn}"
TRAIN_LAUNCH_PID=""

if [[ ! -f "${FASTA_LIST}" ]]; then
  echo "FASTA_LIST does not exist: ${FASTA_LIST}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}"

read -r -a EXPORT_GPUS <<< "${EXPORT_GPUS_STRING}"
if [[ "${#EXPORT_GPUS[@]}" -lt 3 ]]; then
  echo "EXPORT_GPUS needs at least 3 GPU ids, e.g. '2 3 4'" >&2
  exit 1
fi

QFORMER_RUN_DIR="${RUN_ROOT}/128k_qformer_small"
MEAN_RUN_DIR="${RUN_ROOT}/128k_mean_baseline"
QFORMER_CKPT="${QFORMER_RUN_DIR}/checkpoint_epoch_${TRAIN_EPOCHS}.pt"
MEAN_CKPT="${MEAN_RUN_DIR}/checkpoint_epoch_${TRAIN_EPOCHS}.pt"
QFORMER_EVAL_DIR="${EVAL_ROOT}/qformer_small"
MEAN_EVAL_DIR="${EVAL_ROOT}/mean_baseline"

count_manifest_samples() {
  local manifest_path="$1"
  if [[ ! -f "${manifest_path}" ]]; then
    echo 0
    return 0
  fi
  wc -l < "${manifest_path}" | tr -d '[:space:]'
}

count_hidden_state_files() {
  local bucket="$1"
  local split="$2"
  local split_dir="${HS_ROOT}/${bucket}/${split}"
  if [[ ! -d "${split_dir}" ]]; then
    echo 0
    return 0
  fi
  find "${split_dir}" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d '[:space:]'
}

split_complete() {
  local bucket="$1"
  local split="$2"
  local manifest_path="${DATASET_ROOT}/manifests/${bucket}/${split}.jsonl"
  local manifest_count
  local hidden_count
  manifest_count="$(count_manifest_samples "${manifest_path}")"
  if [[ "${manifest_count}" -eq 0 ]]; then
    return 1
  fi
  hidden_count="$(count_hidden_state_files "${bucket}" "${split}")"
  [[ "${hidden_count}" -ge "${manifest_count}" ]]
}

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

echo "[1/5] Prepare manifests"
FASTA_LIST="${FASTA_LIST}" \
OUTPUT_ROOT="${DATASET_ROOT}" \
SEED="${SEED}" \
WORKERS="${WORKERS}" \
SUMMARY_CHUNKSIZE=32 \
EXTRACT_CHUNKSIZE=8 \
PROGRESS_EVERY=250 \
REUSE_METADATA=1 \
SELECTION_MODE=nested \
BUCKET_SPECS="8k:8192:${TRAIN_GENOMES_128K}:${VAL_GENOMES_SHARED}:0 32k:32768:${TRAIN_GENOMES_128K}:${VAL_GENOMES_SHARED}:0 128k:131072:${TRAIN_GENOMES_128K}:${VAL_GENOMES_SHARED}:0" \
WINDOWS_PER_GENOME="8k:0:1:0 32k:0:1:0 128k:${WINDOWS_128K_TRAIN}:${WINDOWS_128K_VAL}:0" \
bash "${PROJECT_ROOT}/scripts/01_prepare_feasibility_data.sh" \
  > "${LOG_ROOT}/01_prepare.log" 2>&1

echo "[2/5] Extract hidden states"
if split_complete 128k train && split_complete 8k val && split_complete 32k val && split_complete 128k val; then
  echo "Skip extraction: required hidden-state splits are already complete." \
    > "${LOG_ROOT}/02_extract_driver.log"
else
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  HS_ROOT="${HS_ROOT}" \
  GPU_IDS="${EXTRACT_GPU_IDS}" \
  MODEL_DTYPE=bfloat16 \
  SAVE_DTYPE=bfloat16 \
  LAUNCHER=python \
  LOG_ROOT="${LOG_ROOT}/extract" \
  bash "${PROJECT_ROOT}/scripts/02_extract_hidden_states.sh" \
    > "${LOG_ROOT}/02_extract_driver.log" 2>&1
fi

echo "[3/5] Train 128k qformer_small and mean baseline"
TRAIN_PIDS=()
if [[ ! -f "${QFORMER_CKPT}" ]]; then
  launch_train_job \
    "${TRAIN_GPU_IDS_QFORMER}" \
    "${TRAIN_MASTER_PORT_QFORMER}" \
    "${LOG_ROOT}/03_train_qformer.log" \
    "${PROJECT_ROOT}/train.py" \
    --config "${PROJECT_ROOT}/configs/v1.yaml" \
    --output-dir "${QFORMER_RUN_DIR}" \
    --override \
      "data.data_root=${HS_ROOT}/128k/train" \
      "data.hidden_dtype=bfloat16" \
      "data.max_length=131072" \
      "data.random_crop=false" \
      "data.num_workers=${TRAIN_NUM_WORKERS}" \
      "training.batch_size=${TRAIN_BATCH_SIZE}" \
      "training.epochs=${TRAIN_EPOCHS}" \
      "corruption.mask_ratio=${MASK_RATIO}" \
      "corruption.mask_target_span=true" \
      "loss.variance_weight=${VARIANCE_WEIGHT}" \
      "loss.covariance_weight=${COVARIANCE_WEIGHT}" \
      "model.type=qformer" \
      "model.num_queries=${QFORMER_NUM_QUERIES}" \
      "model.num_layers=${QFORMER_NUM_LAYERS}" \
      "model.seq_pool=${QFORMER_SEQ_POOL}"
  TRAIN_PIDS+=("${TRAIN_LAUNCH_PID}")
else
  echo "Skip qformer training: checkpoint already exists at ${QFORMER_CKPT}" \
    > "${LOG_ROOT}/03_train_qformer.log"
fi

if [[ ! -f "${MEAN_CKPT}" ]]; then
  launch_train_job \
    "${TRAIN_GPU_IDS_MEAN}" \
    "${TRAIN_MASTER_PORT_MEAN}" \
    "${LOG_ROOT}/03_train_mean.log" \
    "${PROJECT_ROOT}/train.py" \
    --config "${PROJECT_ROOT}/configs/v1.yaml" \
    --output-dir "${MEAN_RUN_DIR}" \
    --override \
      "data.data_root=${HS_ROOT}/128k/train" \
      "data.hidden_dtype=bfloat16" \
      "data.max_length=131072" \
      "data.random_crop=false" \
      "data.num_workers=${TRAIN_NUM_WORKERS}" \
      "training.batch_size=${TRAIN_BATCH_SIZE}" \
      "training.epochs=${TRAIN_EPOCHS}" \
      "corruption.mask_ratio=${MASK_RATIO}" \
      "corruption.mask_target_span=true" \
      "loss.variance_weight=${VARIANCE_WEIGHT}" \
      "loss.covariance_weight=${COVARIANCE_WEIGHT}" \
      "model.type=mean"
  TRAIN_PIDS+=("${TRAIN_LAUNCH_PID}")
else
  echo "Skip mean training: checkpoint already exists at ${MEAN_CKPT}" \
    > "${LOG_ROOT}/03_train_mean.log"
fi

if [[ "${#TRAIN_PIDS[@]}" -gt 0 ]]; then
  wait "${TRAIN_PIDS[@]}"
fi

bucket_max_length() {
  case "$1" in
    8k) echo 8192 ;;
    32k) echo 32768 ;;
    128k) echo 131072 ;;
    *) echo "Unknown bucket $1" >&2; return 1 ;;
  esac
}

export_view() {
  local checkpoint_path="$1"
  local bucket="$2"
  local gpu="$3"
  local output_path="$4"
  local max_length
  max_length="$(bucket_max_length "${bucket}")"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${PROJECT_ROOT}/eval/export_embeddings.py" \
    --config "${PROJECT_ROOT}/configs/v1.yaml" \
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

echo "[4/5] Export val embeddings"
mkdir -p "${QFORMER_EVAL_DIR}" "${MEAN_EVAL_DIR}"

EXPORT_PIDS=()
if [[ ! -f "${QFORMER_EVAL_DIR}/8k_embeddings.pt" ]]; then
  export_view "${QFORMER_CKPT}" 8k "${EXPORT_GPUS[0]}" "${QFORMER_EVAL_DIR}/8k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ ! -f "${QFORMER_EVAL_DIR}/32k_embeddings.pt" ]]; then
  export_view "${QFORMER_CKPT}" 32k "${EXPORT_GPUS[1]}" "${QFORMER_EVAL_DIR}/32k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ ! -f "${QFORMER_EVAL_DIR}/128k_embeddings.pt" ]]; then
  export_view "${QFORMER_CKPT}" 128k "${EXPORT_GPUS[2]}" "${QFORMER_EVAL_DIR}/128k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ "${#EXPORT_PIDS[@]}" -gt 0 ]]; then
  wait "${EXPORT_PIDS[@]}"
fi

EXPORT_PIDS=()
if [[ ! -f "${MEAN_EVAL_DIR}/8k_embeddings.pt" ]]; then
  export_view "${MEAN_CKPT}" 8k "${EXPORT_GPUS[0]}" "${MEAN_EVAL_DIR}/8k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ ! -f "${MEAN_EVAL_DIR}/32k_embeddings.pt" ]]; then
  export_view "${MEAN_CKPT}" 32k "${EXPORT_GPUS[1]}" "${MEAN_EVAL_DIR}/32k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ ! -f "${MEAN_EVAL_DIR}/128k_embeddings.pt" ]]; then
  export_view "${MEAN_CKPT}" 128k "${EXPORT_GPUS[2]}" "${MEAN_EVAL_DIR}/128k_embeddings.pt" &
  EXPORT_PIDS+=("$!")
fi
if [[ "${#EXPORT_PIDS[@]}" -gt 0 ]]; then
  wait "${EXPORT_PIDS[@]}"
fi

echo "[5/5] Run unsupervised retrieval / consistency"
if [[ ! -f "${QFORMER_EVAL_DIR}/retrieval_consistency.json" ]]; then
  python "${PROJECT_ROOT}/eval/retrieval_consistency.py" \
    --embeddings \
      8k="${QFORMER_EVAL_DIR}/8k_embeddings.pt" \
      32k="${QFORMER_EVAL_DIR}/32k_embeddings.pt" \
      128k="${QFORMER_EVAL_DIR}/128k_embeddings.pt" \
    --output "${QFORMER_EVAL_DIR}/retrieval_consistency.json" \
    > "${LOG_ROOT}/05_eval_qformer.log" 2>&1
else
  echo "Skip qformer retrieval eval: output already exists at ${QFORMER_EVAL_DIR}/retrieval_consistency.json" \
    > "${LOG_ROOT}/05_eval_qformer.log"
fi

if [[ ! -f "${MEAN_EVAL_DIR}/retrieval_consistency.json" ]]; then
  python "${PROJECT_ROOT}/eval/retrieval_consistency.py" \
    --embeddings \
      8k="${MEAN_EVAL_DIR}/8k_embeddings.pt" \
      32k="${MEAN_EVAL_DIR}/32k_embeddings.pt" \
      128k="${MEAN_EVAL_DIR}/128k_embeddings.pt" \
    --output "${MEAN_EVAL_DIR}/retrieval_consistency.json" \
    > "${LOG_ROOT}/05_eval_mean.log" 2>&1
else
  echo "Skip mean retrieval eval: output already exists at ${MEAN_EVAL_DIR}/retrieval_consistency.json" \
    > "${LOG_ROOT}/05_eval_mean.log"
fi

cat > "${OUTPUT_ROOT}/README_run_summary.txt" <<EOF
Run tag: ${RUN_TAG}
FASTA list: ${FASTA_LIST}
Output root: ${OUTPUT_ROOT}

Main training data:
  128k train genomes: ${TRAIN_GENOMES_128K}
  128k train windows/genome: ${WINDOWS_128K_TRAIN}
  shared val genomes across 8k/32k/128k: ${VAL_GENOMES_SHARED}
  qformer GPUs: ${TRAIN_GPU_IDS_QFORMER}
  mean GPUs: ${TRAIN_GPU_IDS_MEAN}
  train batch_size per GPU: ${TRAIN_BATCH_SIZE}
  train data workers per rank: ${TRAIN_NUM_WORKERS}

Training outputs:
  qformer_small: ${QFORMER_RUN_DIR}
  mean_baseline: ${MEAN_RUN_DIR}

Evaluation outputs:
  qformer_small retrieval: ${QFORMER_EVAL_DIR}/retrieval_consistency.json
  mean_baseline retrieval: ${MEAN_EVAL_DIR}/retrieval_consistency.json

Logs:
  ${LOG_ROOT}
EOF

echo "Finished. Summary: ${OUTPUT_ROOT}/README_run_summary.txt"
