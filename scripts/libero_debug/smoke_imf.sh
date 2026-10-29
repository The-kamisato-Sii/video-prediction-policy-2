#!/usr/bin/env bash
# Diagnostic only; formal launchers keep global batch 256 automatically.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
method="${1:?imf_boundary or imf_auxiliary}"
case "$method" in imf_boundary|imf_auxiliary) ;; *) exit 2 ;; esac
source scripts/libero/activate_env.sh
source scripts/libero/topology.sh
export PYTHONPATH="$repo/checks:$PYTHONPATH"
export VPP2_TRAIN_MODULE=libero_imf_smoke VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export VPP2_CODE_REVISION="$(git rev-parse HEAD)"
export VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1 OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" WANDB_MODE=offline
export VPP2_SMOKE_STEPS="${VPP2_SMOKE_STEPS:-10}"
name="${RUN_NAME:-libero-${method}-${TRAIN_JOB_ID:-manual}-round${RUNNING_ROUND:-0}}-smoke"
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2
mkdir -p "$repo/runs/libero/${name}_logs"
bash scripts/train.sh --benchmark libero --config "configs/libero/train_${method}.yaml" --allow-batch-change \
  "output_dir=$repo/runs/libero/$name" overwrite=false batch_size=16 num_workers=4 log_every=1 \
  wandb.enabled=true wandb.mode=offline wandb.project=VPP2-LIBERO-imf-smoke "wandb.name=$name" \
  > >(tee -a "$repo/runs/libero/${name}_logs/node${NODE_RANK}.log") 2>&1
