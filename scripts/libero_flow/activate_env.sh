#!/usr/bin/env bash
# Isolated dependency overlay; never pip-install into the running baseline env.
flow_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$flow_repo/scripts/libero/activate_env.sh"
export PYTHON_BIN="$flow_repo/runs/libero_flow_runtime/venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo 'Run scripts/libero_flow/setup.sh first.' >&2; return 1; }
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
unset flow_repo
