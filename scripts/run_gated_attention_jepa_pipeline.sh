#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/outputs/hours_unlabeled_128k/dataset}"
HS_ROOT="${HS_ROOT:-$DATASET_ROOT/hidden_states}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-2-5B-8k}"

RUN_TAG="${RUN_TAG:-gated_attention_jepa}"
RUN_NAME="${RUN_NAME:-gated_attention_cov_5e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_ROOT/runs}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"

TRAIN_GPU_GROUP="${TRAIN_GPU_GROUP:-0}"
MASTER_PORT="${MASTER_PORT:-29810}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"
MASK_RATIO="${MASK_RATIO:-0.30}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-0.01}"
COVARIANCE_WEIGHT="${COVARIANCE_WEIGHT:-5e-5}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
CHUNK_STRIDE="${CHUNK_STRIDE:-64}"

TRAIN_JSONL="${TRAIN_JSONL:-}"
VALID_JSONL="${VALID_JSONL:-}"
TEST_JSONL="${TEST_JSONL:-}"
DATA_DIR="${DATA_DIR:-}"
INPUT_ID_KEY="${INPUT_ID_KEY:-id}"
INPUT_SEQUENCE_KEY="${INPUT_SEQUENCE_KEY:-seq}"
INPUT_LABEL_KEY="${INPUT_LABEL_KEY:-label}"
TASK="${TASK:-auto}"

EXPORT_GPUS_STRING="${EXPORT_GPUS:-0 1 2}"
PROBE_GPU="${PROBE_GPU:-0}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-64}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-0}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

PROBE_CLASSIFIERS="${PROBE_CLASSIFIERS:-linear xgboost}"
PROBE_SEEDS="${PROBE_SEEDS:-41 42 43 44 45}"
PROBE_EPOCHS="${PROBE_EPOCHS:-100}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_LR="${PROBE_LR:-1e-3}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0}"
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-300}"
XGB_LEARNING_RATE="${XGB_LEARNING_RATE:-0.05}"
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-6}"
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8}"
XGB_COLSAMPLE_BYTREE="${XGB_COLSAMPLE_BYTREE:-0.8}"
XGB_REG_LAMBDA="${XGB_REG_LAMBDA:-1.0}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-20}"
XGB_TREE_METHOD="${XGB_TREE_METHOD:-hist}"
XGB_DEVICE="${XGB_DEVICE:-auto}"
XGB_N_JOBS="${XGB_N_JOBS:-0}"

if [[ -n "${DATA_DIR}" ]]; then
  TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train.jsonl}"
  if [[ -z "${VALID_JSONL}" ]]; then
    if [[ -f "${DATA_DIR}/valid.jsonl" ]]; then
      VALID_JSONL="${DATA_DIR}/valid.jsonl"
    elif [[ -f "${DATA_DIR}/validation.jsonl" ]]; then
      VALID_JSONL="${DATA_DIR}/validation.jsonl"
    elif [[ -f "${DATA_DIR}/eval.jsonl" ]]; then
      VALID_JSONL="${DATA_DIR}/eval.jsonl"
    fi
  fi
  TEST_JSONL="${TEST_JSONL:-${DATA_DIR}/test.jsonl}"
fi

mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"

if [[ ! -d "${HS_ROOT}/128k/train" ]]; then
  echo "Missing training hidden states: ${HS_ROOT}/128k/train" >&2
  exit 1
fi

echo "[1/3] Train gated-attention JEPA"
CHECKPOINT_PATH="${RUN_DIR}/checkpoint_epoch_${TRAIN_EPOCHS}.pt"
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  echo "Skip training: checkpoint exists at ${CHECKPOINT_PATH}"
else
  if [[ "${TRAIN_GPU_GROUP}" == *" "* ]]; then
    IFS=' ' read -r -a GPU_IDS <<< "${TRAIN_GPU_GROUP}"
    WORLD_SIZE="${#GPU_IDS[@]}"
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")" \
      torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${WORLD_SIZE}" \
        --master_port="${MASTER_PORT}" \
        "${PROJECT_ROOT}/train.py" \
        --config "${PROJECT_ROOT}/configs/gated_attention_jepa.yaml" \
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
          "loss.covariance_weight=${COVARIANCE_WEIGHT}" \
          "model.type=gated_attention" \
          "model.gate_hidden_dim=${GATE_HIDDEN_DIM}" \
          "model.chunk_pooling.chunk_size=${CHUNK_SIZE}" \
          "model.chunk_pooling.stride=${CHUNK_STRIDE}" \
      > "${LOG_ROOT}/train_${RUN_NAME}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_GROUP}" \
      python "${PROJECT_ROOT}/train.py" \
        --config "${PROJECT_ROOT}/configs/gated_attention_jepa.yaml" \
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
          "loss.covariance_weight=${COVARIANCE_WEIGHT}" \
          "model.type=gated_attention" \
          "model.gate_hidden_dim=${GATE_HIDDEN_DIM}" \
          "model.chunk_pooling.chunk_size=${CHUNK_SIZE}" \
          "model.chunk_pooling.stride=${CHUNK_STRIDE}" \
      > "${LOG_ROOT}/train_${RUN_NAME}.log" 2>&1
  fi
fi

if [[ -z "${TRAIN_JSONL}" || -z "${VALID_JSONL}" || -z "${TEST_JSONL}" ]]; then
  echo "Skip probe: set DATA_DIR or TRAIN_JSONL/VALID_JSONL/TEST_JSONL to run downstream evaluation."
  exit 0
fi

for JSONL_PATH in "${TRAIN_JSONL}" "${VALID_JSONL}" "${TEST_JSONL}"; do
  if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "Missing JSONL file: ${JSONL_PATH}" >&2
    exit 1
  fi
done

read -r -a EXPORT_GPUS <<< "${EXPORT_GPUS_STRING}"
read -r -a CLASSIFIERS <<< "${PROBE_CLASSIFIERS}"

PROBE_OUTPUT_ROOT="${OUTPUT_ROOT}/probe/${RUN_NAME}"
EMB_ROOT="${PROBE_OUTPUT_ROOT}/embeddings"
RESULT_ROOT="${PROBE_OUTPUT_ROOT}/results"
PROBE_LOG_ROOT="${PROBE_OUTPUT_ROOT}/logs"
mkdir -p "${EMB_ROOT}" "${RESULT_ROOT}" "${PROBE_LOG_ROOT}"

echo "[2/3] Export split embeddings"
SPLITS=(train validation test)
PIDS=()
for idx in "${!SPLITS[@]}"; do
  SPLIT="${SPLITS[$idx]}"
  case "${SPLIT}" in
    train) JSONL_PATH="${TRAIN_JSONL}" ;;
    validation) JSONL_PATH="${VALID_JSONL}" ;;
    test) JSONL_PATH="${TEST_JSONL}" ;;
  esac

  EMB_PATH="${EMB_ROOT}/${SPLIT}_embeddings.pt"
  if [[ -f "${EMB_PATH}" ]]; then
    echo "Skip existing embedding: ${EMB_PATH}"
    continue
  fi

  GPU="${EXPORT_GPUS[$((idx % ${#EXPORT_GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${PROJECT_ROOT}/eval/export_embeddings_from_sequences.py" \
    --config "${RUN_DIR}/resolved_config.json" \
    --model-path "${MODEL_PATH}" \
    --manifest "${JSONL_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output "${EMB_PATH}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --num-workers "${EXPORT_NUM_WORKERS}" \
    --sequence-key "${INPUT_SEQUENCE_KEY}" \
    --id-key "${INPUT_ID_KEY}" \
    --label-key "${INPUT_LABEL_KEY}" \
    --model-dtype "${MODEL_DTYPE}" \
    > "${PROBE_LOG_ROOT}/export_${SPLIT}.log" 2>&1 &
  PIDS+=("$!")
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait "${PIDS[@]}"
fi

echo "[3/3] Run split probes"
for CLASSIFIER in "${CLASSIFIERS[@]}"; do
  CUDA_VISIBLE_DEVICES="${PROBE_GPU}" python "${PROJECT_ROOT}/eval/split_linear_probe.py" \
    --train-embeddings "${EMB_ROOT}/train_embeddings.pt" \
    --val-embeddings "${EMB_ROOT}/validation_embeddings.pt" \
    --test-embeddings "${EMB_ROOT}/test_embeddings.pt" \
    --task "${TASK}" \
    --classifier "${CLASSIFIER}" \
    --seeds ${PROBE_SEEDS} \
    --epochs "${PROBE_EPOCHS}" \
    --batch-size "${PROBE_BATCH_SIZE}" \
    --lr "${PROBE_LR}" \
    --weight-decay "${PROBE_WEIGHT_DECAY}" \
    --xgb-n-estimators "${XGB_N_ESTIMATORS}" \
    --xgb-learning-rate "${XGB_LEARNING_RATE}" \
    --xgb-max-depth "${XGB_MAX_DEPTH}" \
    --xgb-subsample "${XGB_SUBSAMPLE}" \
    --xgb-colsample-bytree "${XGB_COLSAMPLE_BYTREE}" \
    --xgb-reg-lambda "${XGB_REG_LAMBDA}" \
    --xgb-early-stopping-rounds "${XGB_EARLY_STOPPING_ROUNDS}" \
    --xgb-tree-method "${XGB_TREE_METHOD}" \
    --xgb-device "${XGB_DEVICE}" \
    --xgb-n-jobs "${XGB_N_JOBS}" \
    --output "${RESULT_ROOT}/split_${CLASSIFIER}_probe.json" \
    > "${PROBE_LOG_ROOT}/probe_${CLASSIFIER}.log" 2>&1
done

echo "Finished. Run dir: ${RUN_DIR}"
echo "Probe dir: ${PROBE_OUTPUT_ROOT}"
