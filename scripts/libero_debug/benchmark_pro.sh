#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source scripts/libero/activate_pro_env.sh
output_dir="$repo/runs/libero_pro_2gpu/${TRAIN_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%S)}-round${RUNNING_ROUND:-0}"
exec "$PYTHON_BIN" -u scripts/libero_debug/benchmark_pro.py --output-dir "$output_dir"
