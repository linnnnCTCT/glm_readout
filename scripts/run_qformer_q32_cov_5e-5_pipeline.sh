#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FASTA_LIST="${FASTA_LIST:-$PROJECT_ROOT/MAGsANI97.txt}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/model/Genos_m.GQA-MoE32-5B-32k}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/outputs/qformer_q32_cov_5e-5_curriculum_dataset}"
HS_ROOT="${HS_ROOT:-$DATASET_ROOT/hidden_states}"
RUN_TAG="${RUN_TAG:-qformer_q32_cov_5e-5_dout1024_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/${RUN_TAG}}"

LABELS_TSV="${LABELS_TSV:-}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0 1 2 3}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-bfloat16}"

SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"
CONTIG_SPACER_CHAR="${CONTIG_SPACER_CHAR:-#}"
CONTIG_SPACER_LENGTH="${CONTIG_SPACER_LENGTH:-1}"
SELECTION_MODE="${SELECTION_MODE:-nested}"
REUSE_METADATA="${REUSE_METADATA:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-contextagg-readout}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

BUCKET_SPECS="${BUCKET_SPECS:-8k:8192:3000:400:400 32k:32768:1500:200:200 128k:131072:300:50:50 1m:1048576:40:10:10}"
WINDOWS_PER_GENOME="${WINDOWS_PER_GENOME:-8k:2:2:2 32k:2:2:2 128k:1:1:1 1m:1:1:1}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

if [[ ! -f "${FASTA_LIST}" ]]; then
  echo "FASTA_LIST does not exist: ${FASTA_LIST}" >&2
  exit 1
fi

mkdir -p "${DATASET_ROOT}" "${OUTPUT_ROOT}"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  echo "[1/3] Prepare manifests from FASTA list"
  FASTA_LIST="${FASTA_LIST}" \
  OUTPUT_ROOT="${DATASET_ROOT}" \
  LABELS_TSV="${LABELS_TSV}" \
  SEED="${SEED}" \
  WORKERS="${WORKERS}" \
  REUSE_METADATA="${REUSE_METADATA}" \
  SELECTION_MODE="${SELECTION_MODE}" \
  CONTIG_SPACER_CHAR="${CONTIG_SPACER_CHAR}" \
  CONTIG_SPACER_LENGTH="${CONTIG_SPACER_LENGTH}" \
  BUCKET_SPECS="${BUCKET_SPECS}" \
  WINDOWS_PER_GENOME="${WINDOWS_PER_GENOME}" \
  bash "${PROJECT_ROOT}/scripts/01_prepare_feasibility_data.sh"
fi

if [[ "${RUN_EXTRACT}" == "1" ]]; then
  echo "[2/3] Extract hidden states"
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  HS_ROOT="${HS_ROOT}" \
  MODEL_DTYPE="${MODEL_DTYPE}" \
  SAVE_DTYPE="${SAVE_DTYPE}" \
  GPU_IDS="${EXTRACT_GPUS}" \
  bash "${PROJECT_ROOT}/scripts/02_extract_hidden_states.sh"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[3/3] Run curriculum training"
  HS_ROOT="${HS_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  TRAIN_GPUS="${TRAIN_GPUS}" \
  TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS}" \
  TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR}" \
  WANDB_ENABLED="${WANDB_ENABLED}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  bash "${PROJECT_ROOT}/scripts/run_qformer_q32_cov_5e-5_curriculum.sh"
fi

cat > "${OUTPUT_ROOT}/README_pipeline.txt" <<EOF
FASTA list: ${FASTA_LIST}
Model path: ${MODEL_PATH}
Dataset root: ${DATASET_ROOT}
Hidden-state root: ${HS_ROOT}
Run tag: ${RUN_TAG}
Output root: ${OUTPUT_ROOT}

Prepare:
  buckets=${BUCKET_SPECS}
  windows_per_genome=${WINDOWS_PER_GENOME}
  selection_mode=${SELECTION_MODE}
  contig_spacer='${CONTIG_SPACER_CHAR}' x ${CONTIG_SPACER_LENGTH}

Extract:
  model_dtype=${MODEL_DTYPE}
  save_dtype=${SAVE_DTYPE}
  extract_gpus=${EXTRACT_GPUS}

Train:
  train_gpus=${TRAIN_GPUS}
  train_num_workers=${TRAIN_NUM_WORKERS}
  train_prefetch_factor=${TRAIN_PREFETCH_FACTOR}
  wandb_enabled=${WANDB_ENABLED}
  wandb_project=${WANDB_PROJECT}
  wandb_entity=${WANDB_ENTITY}
  wandb_mode=${WANDB_MODE}
EOF

echo "Pipeline finished."
