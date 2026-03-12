#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_TSV="${TRAIN_TSV:-$PROJECT_ROOT/data/GTDB1w3.test.set1.tsv}"
TEST_TSV="${TEST_TSV:-$PROJECT_ROOT/data/GTDB1w3.test.set1.100.tsv}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-5B-32k}"
READOUT_CONFIG="${READOUT_CONFIG:-$PROJECT_ROOT/configs/qformer_q16_cov_5e-5_curriculum.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_TAG="${RUN_TAG:-gtdb_species_13k_fewshot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"

EXPORT_GPUS_STRING="${EXPORT_GPUS:-${EXPORT_GPU:-0}}"
PROBE_GPU="${PROBE_GPU:-0}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-0}"

PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"
PROBE_EVAL_BATCH_SIZE="${PROBE_EVAL_BATCH_SIZE:-8192}"
PROBE_LR="${PROBE_LR:-1e-2}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
SELECTION_SEED="${SELECTION_SEED:-42}"
SHOTS="${SHOTS:-1 5 10 20 50}"
AUC_NUM_CLASSES="${AUC_NUM_CLASSES:-1024}"
AUC_SAMPLES_PER_CLASS="${AUC_SAMPLES_PER_CLASS:-8}"
CLUSTER_BATCH_SIZE="${CLUSTER_BATCH_SIZE:-65536}"
CLUSTER_FIT_PASSES="${CLUSTER_FIT_PASSES:-1}"
CLUSTER_N_INIT="${CLUSTER_N_INIT:-3}"

EMB_ROOT="${OUTPUT_ROOT}/embeddings"
RESULT_ROOT="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"
LABEL_MAP_PATH="${OUTPUT_ROOT}/label_map.json"
TRAIN_EMB_PATH="${EMB_ROOT}/train_embeddings.pt"
TEST_EMB_PATH="${EMB_ROOT}/test_embeddings.pt"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH=/abs/path/to/checkpoint.pt" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_TSV}" ]]; then
  echo "Missing TRAIN_TSV: ${TRAIN_TSV}" >&2
  exit 1
fi
if [[ ! -f "${TEST_TSV}" ]]; then
  echo "Missing TEST_TSV: ${TEST_TSV}" >&2
  exit 1
fi

mkdir -p "${EMB_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"
read -r -a EXPORT_GPUS <<< "${EXPORT_GPUS_STRING}"
if [[ "${#EXPORT_GPUS[@]}" -eq 0 ]]; then
  echo "EXPORT_GPUS must contain at least one GPU id." >&2
  exit 1
fi

run_parallel_export() {
  local input_tsv="$1"
  local output_path="$2"
  local label_map_arg_name="$3"
  local label_map_arg_value="$4"
  local log_prefix="$5"

  local num_shards="${#EXPORT_GPUS[@]}"
  local shard_paths=()
  local pids=()

  if [[ "${num_shards}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${EXPORT_GPUS[0]}" python "${PROJECT_ROOT}/eval/export_embeddings_from_species_tsv.py" \
      --config "${READOUT_CONFIG}" \
      --model-path "${MODEL_PATH}" \
      --input-tsv "${input_tsv}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --output "${output_path}" \
      "${label_map_arg_name}" "${label_map_arg_value}" \
      --batch-size "${EXPORT_BATCH_SIZE}" \
      --num-workers "${EXPORT_NUM_WORKERS}" \
      --max-length "${MAX_LENGTH}" \
      --model-dtype "${MODEL_DTYPE}" \
      --save-dtype "${SAVE_DTYPE}" \
      > "${LOG_ROOT}/${log_prefix}.log" 2>&1
    return
  fi

  for shard_index in "${!EXPORT_GPUS[@]}"; do
    local gpu="${EXPORT_GPUS[$shard_index]}"
    local shard_path="${output_path%.pt}.shard${shard_index}.pt"
    shard_paths+=("${shard_path}")
    CUDA_VISIBLE_DEVICES="${gpu}" python "${PROJECT_ROOT}/eval/export_embeddings_from_species_tsv.py" \
      --config "${READOUT_CONFIG}" \
      --model-path "${MODEL_PATH}" \
      --input-tsv "${input_tsv}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --output "${shard_path}" \
      "${label_map_arg_name}" "${label_map_arg_value}" \
      --batch-size "${EXPORT_BATCH_SIZE}" \
      --num-workers "${EXPORT_NUM_WORKERS}" \
      --max-length "${MAX_LENGTH}" \
      --model-dtype "${MODEL_DTYPE}" \
      --save-dtype "${SAVE_DTYPE}" \
      --shard-index "${shard_index}" \
      --num-shards "${num_shards}" \
      > "${LOG_ROOT}/${log_prefix}.shard${shard_index}.log" 2>&1 &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait "${pids[@]}"
  fi

  python "${PROJECT_ROOT}/eval/merge_exported_embeddings.py" \
    --inputs "${shard_paths[@]}" \
    --output "${output_path}" \
    > "${LOG_ROOT}/${log_prefix}.merge.log" 2>&1
}

echo "[1/3] Export train embeddings"
if [[ ! -f "${TRAIN_EMB_PATH}" ]]; then
  run_parallel_export "${TRAIN_TSV}" "${TRAIN_EMB_PATH}" --label-map-out "${LABEL_MAP_PATH}" "01_export_train"
else
  echo "Skip existing file: ${TRAIN_EMB_PATH}"
fi

echo "[2/3] Export test embeddings"
if [[ ! -f "${TEST_EMB_PATH}" ]]; then
  run_parallel_export "${TEST_TSV}" "${TEST_EMB_PATH}" --label-map-in "${LABEL_MAP_PATH}" "02_export_test"
else
  echo "Skip existing file: ${TEST_EMB_PATH}"
fi

echo "[3/3] Run nested few-shot classification and k-means clustering"
CUDA_VISIBLE_DEVICES="${PROBE_GPU}" python "${PROJECT_ROOT}/eval/gtdb_species_fewshot.py" \
  --train-embeddings "${TRAIN_EMB_PATH}" \
  --test-embeddings "${TEST_EMB_PATH}" \
  --output-dir "${RESULT_ROOT}" \
  --shots ${SHOTS} \
  --base-shot 50 \
  --selection-seed "${SELECTION_SEED}" \
  --epochs "${PROBE_EPOCHS}" \
  --batch-size "${PROBE_BATCH_SIZE}" \
  --eval-batch-size "${PROBE_EVAL_BATCH_SIZE}" \
  --lr "${PROBE_LR}" \
  --weight-decay "${PROBE_WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --device auto \
  --auc-num-classes "${AUC_NUM_CLASSES}" \
  --auc-samples-per-class "${AUC_SAMPLES_PER_CLASS}" \
  --cluster-batch-size "${CLUSTER_BATCH_SIZE}" \
  --cluster-fit-passes "${CLUSTER_FIT_PASSES}" \
  --cluster-n-init "${CLUSTER_N_INIT}" \
  > "${LOG_ROOT}/03_fewshot_and_cluster.log" 2>&1

echo "Done. Results: ${RESULT_ROOT}/results.json"
