#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source scripts/libero/activate_env.sh
"$PYTHON_BIN" -m venv --system-site-packages runs/libero_flow_runtime/venv
runs/libero_flow_runtime/venv/bin/python -m pip install --no-deps diffusers==0.31.0 opencv-python-headless==4.11.0.86 lz4==4.3.3
echo 'MotionCrafter weights expected at weights/MotionCrafter/geometry_motion_vae; see README.'
