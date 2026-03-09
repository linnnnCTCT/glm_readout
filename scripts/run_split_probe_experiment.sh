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
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SEQUENCE_EXPORT_GPUS_STRING="${SEQUENCE_EXPORT_GPUS:-${EXTRACT_GPUS:-0 1 2}}"
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

mkdir -p "${OUTPUT_ROOT}" "${DATASET_ROOT}" "${EMB_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

read -r -a SEQUENCE_EXPORT_GPUS <<< "${SEQUENCE_EXPORT_GPUS_STRING}"
if [[ "${#SEQUENCE_EXPORT_GPUS[@]}" -eq 0 ]]; then
  echo "SEQUENCE_EXPORT_GPUS must contain at least one GPU id." >&2
  exit 1
fi

echo "[1/3] Prepare split manifests from TSV"
python "${PROJECT_ROOT}/scripts/prepare_probe_tsv.py" \
  --input "${PROBE_TSV}" \
  --output-dir "${DATASET_ROOT}" \
  > "${LOG_ROOT}/01_prepare_probe_manifests.log" 2>&1

echo "[2/3] Export split embeddings directly from sequences"
SPLITS=(train validation test)
PIDS=()
JOB_INDEX=0
for SPLIT in "${SPLITS[@]}"; do
  MANIFEST_PATH="${MANIFEST_ROOT}/${SPLIT}.jsonl"
  EMB_PATH="${EMB_ROOT}/${SPLIT}_embeddings.pt"
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Skip missing split: ${SPLIT}" >> "${LOG_ROOT}/02_export_probe.log"
    continue
  fi
  if [[ -f "${EMB_PATH}" ]]; then
    echo "Skip completed split export: ${SPLIT}" >> "${LOG_ROOT}/02_export_probe.log"
    continue
  fi

  GPU="${SEQUENCE_EXPORT_GPUS[$((JOB_INDEX % ${#SEQUENCE_EXPORT_GPUS[@]}))]}"
  LOG_FILE="${LOG_ROOT}/export_${SPLIT}.log"
  echo "Launch export: split=${SPLIT} gpu=${GPU}" >> "${LOG_ROOT}/02_export_probe.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_ROOT}/eval/export_embeddings_from_sequences.py" \
    --config "${READOUT_CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --manifest "${MANIFEST_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${EMB_PATH}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --num-workers "${EXPORT_NUM_WORKERS}" \
    --model-dtype "${MODEL_DTYPE}" \
    > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
  JOB_INDEX=$((JOB_INDEX + 1))
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait "${PIDS[@]}"
fi

echo "[3/3] Run split linear probe"
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
  > "${LOG_ROOT}/03_split_linear_probe.log" 2>&1

echo "Finished. Probe result: ${RESULT_ROOT}/split_linear_probe.json"
