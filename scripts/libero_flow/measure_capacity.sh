#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source scripts/libero_flow/activate_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "${FLOW_CAPACITY_BACKEND:-zero1}" in
  zero1)
    exec "$PYTHON_BIN" -m accelerate.commands.launch \
      --config_file scripts/libero_flow/accelerate_zero1.yaml --num_processes 1 \
      --main_process_port "${MASTER_PORT:-29639}" \
      --module vpp2.libero_flow.capacity "$@"
    ;;
  plain)
    exec "$PYTHON_BIN" -m vpp2.libero_flow.capacity "$@"
    ;;
  *) echo 'FLOW_CAPACITY_BACKEND must be zero1 or plain' >&2; exit 2 ;;
esac
