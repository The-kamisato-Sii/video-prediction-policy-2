#!/usr/bin/env bash
# Source this file from any working directory.
_vpp2_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_vpp2_root="$(dirname "$_vpp2_repo")"
source "$_vpp2_root/miniconda/etc/profile.d/conda.sh"
conda activate "$_vpp2_root/miniconda/envs/vpp2"
export PYTHON_BIN="$_vpp2_root/miniconda/envs/vpp2/bin/python"
export CUDA_HOME="$_vpp2_root/tools/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$_vpp2_repo/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset _vpp2_repo _vpp2_root
