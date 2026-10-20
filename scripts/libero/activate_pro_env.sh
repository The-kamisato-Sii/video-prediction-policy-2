#!/usr/bin/env bash
# Source this file; keep PRO simulator dependencies separate from training.
set -euo pipefail
vpp2_pro_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$vpp2_pro_repo/../miniconda/etc/profile.d/conda.sh"
conda activate "$vpp2_pro_repo/../miniconda/envs/vpp2-libero-pro"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$vpp2_pro_repo/../tools/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$vpp2_pro_repo/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
vpp2_pro_glvnd="$vpp2_pro_repo/../tools/libero-pro-glvnd/lib"
if [[ -f "$vpp2_pro_glvnd/libEGL.so.1" ]]; then
    export LD_LIBRARY_PATH="$vpp2_pro_glvnd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
