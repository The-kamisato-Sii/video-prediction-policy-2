#!/usr/bin/env bash
# Preserve the official Action1B recipe; resolve local checkpoint paths explicitly.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export VPP2_TORCH_LOAD_MMAP=1
export VPP2_VIDEO_DECODE_UINT8="${VPP2_VIDEO_DECODE_UINT8:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-29612}"
export VPP2_TRAIN_MODULE="${VPP2_TRAIN_MODULE:-vpp2.libero.cli}"
export VPP2_ACCELERATE_CONFIG="${VPP2_ACCELERATE_CONFIG:-configs/libero/accelerate_zero1.yaml}"
flags=()
overrides=()
for argument in "$@"; do
  case "$argument" in
    --dry-run|--allow-batch-change) flags+=("$argument") ;;
    *) overrides+=("$argument") ;;
  esac
done
exec bash scripts/train.sh --config configs/libero/train_action.yaml \
  "${flags[@]}" \
  "model.model_id=$repo_root/weights/Wan2.1-I2V-14B-480P" \
  "model.dit_checkpoint_path=$repo_root/weights/libero/video_step010000.pt" \
  "model.action_dit_pretrained_path=$repo_root/weights/libero/action_init.pt" \
  "batch_size=${BATCH_SIZE:-32}" \
  "gradient_accumulation_steps=${GRAD_ACC:-1}" \
  "num_workers=${NUM_WORKERS:-4}" \
  "${overrides[@]}"
