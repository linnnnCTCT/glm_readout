#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PROJECT_ROOT}/scripts/01_prepare_feasibility_data.sh"
"${PROJECT_ROOT}/scripts/02_extract_hidden_states.sh"
"${PROJECT_ROOT}/scripts/03_train_feasibility_suite.sh"
"${PROJECT_ROOT}/scripts/04_export_and_probe.sh"
