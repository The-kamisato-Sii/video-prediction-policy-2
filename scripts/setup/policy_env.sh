#!/usr/bin/env bash
# Run after conda create -n vpp2 python=3.10 pip and conda activate vpp2.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
if [[ -z "${CUDA_HOME:-}" && -x "$repo_dir/../tools/cuda-12.8/bin/nvcc" ]]; then
  export CUDA_HOME="$(cd "$repo_dir/../tools/cuda-12.8" && pwd)"
fi
"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 "$python_bin" -m pip install -r requirements-train.txt -c environment-cu128.txt
"$python_bin" -m pip check
"$python_bin" -c 'import accelerate, deepspeed; print("Accelerate", accelerate.__version__, "DeepSpeed", deepspeed.__version__)'
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name()); x=torch.randn(64,64,device="cuda",dtype=torch.bfloat16,requires_grad=True); (x@x).float().square().mean().backward(); torch.cuda.synchronize(); print("CUDA bf16 forward/backward passed")'
