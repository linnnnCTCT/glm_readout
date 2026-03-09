#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROBE_TSV="${PROBE_TSV:-}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-2-5B-8k}"
READOUT_CONFIG="${READOUT_CONFIG:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_TAG="${RUN_TAG:-probe_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"

MAX_LENGTH="${MAX_LENGTH:-8192}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"
EXTRACT_GPUS_STRING="${EXTRACT_GPUS:-0 1 2}"
EXPORT_GPU="${EXPORT_GPU:-3}"
PROBE_GPU="${PROBE_GPU:-4}"

EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-64}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-4}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_LR="${PROBE_LR:-1e-3}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0}"
PROBE_SEED="${PROBE_SEED:-42}"

DATASET_ROOT="${OUTPUT_ROOT}/dataset"
MANIFEST_ROOT="${DATASET_ROOT}/manifests"
HS_ROOT="${DATASET_ROOT}/hidden_states"
EMB_ROOT="${OUTPUT_ROOT}/embeddings"
RESULT_ROOT="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ -z "${PROBE_TSV}" ]]; then
  echo "Set PROBE_TSV=/abs/path/to/processed_essential_genes.tsv" >&2
  exit 1
fi
if [[ -z "${READOUT_CONFIG}" ]]; then
  echo "Set READOUT_CONFIG=/abs/path/to/resolved_config.json" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH=/abs/path/to/checkpoint_epoch_x.pt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${HS_ROOT}" "${EMB_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

read -r -a EXTRACT_GPUS <<< "${EXTRACT_GPUS_STRING}"
if [[ "${#EXTRACT_GPUS[@]}" -eq 0 ]]; then
  echo "EXTRACT_GPUS must contain at least one GPU id." >&2
  exit 1
fi

count_manifest_samples() {
  local split="$1"
  local manifest_path="${MANIFEST_ROOT}/${split}.jsonl"
  if [[ ! -f "${manifest_path}" ]]; then
    echo 0
    return 0
  fi
  wc -l < "${manifest_path}" | tr -d '[:space:]'
}

count_hidden_state_files() {
  local split="$1"
  local split_dir="${HS_ROOT}/${split}"
  if [[ ! -d "${split_dir}" ]]; then
    echo 0
    return 0
  fi
  find "${split_dir}" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d '[:space:]'
}

echo "[1/4] Prepare split manifests from TSV"
python "${PROJECT_ROOT}/scripts/prepare_probe_tsv.py" \
  --input "${PROBE_TSV}" \
  --output-dir "${DATASET_ROOT}" \
  > "${LOG_ROOT}/01_prepare_probe_manifests.log" 2>&1

echo "[2/4] Extract hidden states for train/validation/test"
SPLITS=(train validation test)
PIDS=()
JOB_INDEX=0
for SPLIT in "${SPLITS[@]}"; do
  MANIFEST_PATH="${MANIFEST_ROOT}/${SPLIT}.jsonl"
  OUTPUT_DIR="${HS_ROOT}/${SPLIT}"
  mkdir -p "${OUTPUT_DIR}"

  MANIFEST_COUNT="$(count_manifest_samples "${SPLIT}")"
  EXISTING_COUNT="$(count_hidden_state_files "${SPLIT}")"
  if [[ "${MANIFEST_COUNT}" -eq 0 ]]; then
    echo "Skip empty split: ${SPLIT}" >> "${LOG_ROOT}/02_extract_probe.log"
    continue
  fi
  if [[ "${EXISTING_COUNT}" -ge "${MANIFEST_COUNT}" ]]; then
    echo "Skip completed split: ${SPLIT} ${EXISTING_COUNT}/${MANIFEST_COUNT}" >> "${LOG_ROOT}/02_extract_probe.log"
    continue
  fi

  GPU="${EXTRACT_GPUS[$((JOB_INDEX % ${#EXTRACT_GPUS[@]}))]}"
  LOG_FILE="${LOG_ROOT}/extract_${SPLIT}.log"
  echo "Launch extraction: split=${SPLIT} gpu=${GPU}" >> "${LOG_ROOT}/02_extract_probe.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_ROOT}/data/extract_hidden_states.py" \
    --model-path "${MODEL_PATH}" \
    --input "${MANIFEST_PATH}" \
    --input-format jsonl \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EXTRACT_BATCH_SIZE}" \
    --device cuda \
    --model-dtype "${MODEL_DTYPE}" \
    --save-dtype "${SAVE_DTYPE}" \
    --skip-existing \
    > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
  JOB_INDEX=$((JOB_INDEX + 1))
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait "${PIDS[@]}"
fi

echo "[3/4] Export split embeddings"
for SPLIT in "${SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="${EXPORT_GPU}" python "${PROJECT_ROOT}/eval/export_embeddings.py" \
    --config "${READOUT_CONFIG}" \
    --override \
      "data.data_root=${HS_ROOT}/${SPLIT}" \
      "data.hidden_dtype=${SAVE_DTYPE}" \
      "data.max_length=${MAX_LENGTH}" \
      "data.random_crop=false" \
      "data.num_workers=${EXPORT_NUM_WORKERS}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${EMB_ROOT}/${SPLIT}_embeddings.pt" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --num-workers "${EXPORT_NUM_WORKERS}" \
    > "${LOG_ROOT}/export_${SPLIT}.log" 2>&1
done

echo "[4/4] Run split linear probe"
CUDA_VISIBLE_DEVICES="${PROBE_GPU}" python "${PROJECT_ROOT}/eval/split_linear_probe.py" \
  --train-embeddings "${EMB_ROOT}/train_embeddings.pt" \
  --val-embeddings "${EMB_ROOT}/validation_embeddings.pt" \
  --test-embeddings "${EMB_ROOT}/test_embeddings.pt" \
  --epochs "${PROBE_EPOCHS}" \
  --lr "${PROBE_LR}" \
  --batch-size "${PROBE_BATCH_SIZE}" \
  --weight-decay "${PROBE_WEIGHT_DECAY}" \
  --seed "${PROBE_SEED}" \
  --output "${RESULT_ROOT}/split_linear_probe.json" \
  > "${LOG_ROOT}/04_split_linear_probe.log" 2>&1

echo "Finished. Probe result: ${RESULT_ROOT}/split_linear_probe.json"
