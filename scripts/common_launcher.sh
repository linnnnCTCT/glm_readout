#!/usr/bin/env bash

set -euo pipefail

LAUNCH_PID=""

is_python_command() {
  local entrypoint="${1:-}"
  [[ "$entrypoint" == "python" || "$entrypoint" == "python3" ]]
}


is_shell_command() {
  local entrypoint="${1:-}"
  [[ "$entrypoint" == "bash" || "$entrypoint" == "sh" || "$entrypoint" == "zsh" ]]
}

launch_single_gpu_job() {
  local launcher="$1"
  local gpu="$2"
  local log_file="$3"
  local master_port="$4"
  shift 4
  local -a cmd=("$@")

  mkdir -p "$(dirname "$log_file")"

  case "$launcher" in
    python)
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_file" 2>&1 &
      ;;
    torchrun)
      if is_python_command "${cmd[0]:-}"; then
        cmd=("${cmd[@]:1}")
      elif is_shell_command "${cmd[0]:-}"; then
        echo "[launcher] shell command detected; bypass torchrun and launch directly." >"$log_file"
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >>"$log_file" 2>&1 &
        LAUNCH_PID="$!"
        return 0
      fi

      CUDA_VISIBLE_DEVICES="$gpu" torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=1 \
        --master_port "$master_port" \
        "${cmd[@]}" >"$log_file" 2>&1 &
      ;;
    deepspeed)
      if is_python_command "${cmd[0]:-}"; then
        cmd=("${cmd[@]:1}")
      elif is_shell_command "${cmd[0]:-}"; then
        echo "[launcher] shell command detected; bypass deepspeed and launch directly." >"$log_file"
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >>"$log_file" 2>&1 &
        LAUNCH_PID="$!"
        return 0
      fi

      CUDA_VISIBLE_DEVICES="$gpu" deepspeed \
        --master_port "$master_port" \
        --num_gpus 1 \
        "${cmd[@]}" >"$log_file" 2>&1 &
      ;;
    *)
      echo "Unsupported launcher: $launcher" >&2
      return 1
      ;;
  esac

  LAUNCH_PID="$!"
}


wait_for_pids() {
  local pids=("$@")
  local pid
  for pid in "${pids[@]}"; do
    if [[ -z "${pid}" ]]; then
      continue
    fi
    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
      echo "[launcher] skip invalid pid '${pid}'" >&2
      continue
    fi
    wait "$pid"
  done
}
