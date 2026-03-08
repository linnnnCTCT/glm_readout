#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FASTA_LIST="${FASTA_LIST:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/feasibility_dataset}"
LABELS_TSV="${LABELS_TSV:-}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
SUMMARY_CHUNKSIZE="${SUMMARY_CHUNKSIZE:-32}"
EXTRACT_CHUNKSIZE="${EXTRACT_CHUNKSIZE:-8}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
REUSE_METADATA="${REUSE_METADATA:-1}"
SELECTION_MODE="${SELECTION_MODE:-independent}"

read -r -a BUCKET_SPECS <<< "${BUCKET_SPECS:-8k:8192:3000:400:400 32k:32768:1500:200:200 128k:131072:300:50:50 1m:1048576:40:10:10}"
read -r -a WINDOWS_PER_GENOME_SPECS <<< "${WINDOWS_PER_GENOME:-8k:4:1:1 32k:4:1:1 128k:2:1:1 1m:1:1:1}"

if [[ -z "${FASTA_LIST}" ]]; then
  echo "Set FASTA_LIST=/abs/path/to/fasta_files.txt" >&2
  exit 1
fi

ARGS=(
  "$PROJECT_ROOT/scripts/prepare_fasta_windows.py"
  --fasta-list "$FASTA_LIST"
  --output-dir "$OUTPUT_ROOT"
  --seed "$SEED"
  --concat-contigs
  --contig-spacer-length 256
  --workers "$WORKERS"
  --summary-chunksize "$SUMMARY_CHUNKSIZE"
  --extract-chunksize "$EXTRACT_CHUNKSIZE"
  --progress-every "$PROGRESS_EVERY"
  --selection-mode "$SELECTION_MODE"
  --buckets
  "${BUCKET_SPECS[@]}"
  --windows-per-genome
  "${WINDOWS_PER_GENOME_SPECS[@]}"
)

if [[ -n "${LABELS_TSV}" ]]; then
  ARGS+=(--labels-tsv "$LABELS_TSV")
fi

if [[ "${REUSE_METADATA}" == "1" ]]; then
  ARGS+=(--reuse-metadata)
fi

python "${ARGS[@]}"

echo "Prepared manifests under: $OUTPUT_ROOT"
