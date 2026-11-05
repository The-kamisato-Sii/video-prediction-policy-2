#!/usr/bin/env bash
# Convert completed GT only; --watch also follows newly finished episodes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source scripts/libero_flow/activate_env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 10 "$PYTHON_BIN" -u -m vpp2.libero_flow.convert_motions "$@"
