#!/usr/bin/env bash
# Shared one-step launcher with automatic global-batch allocation.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
method="${1:?Specify a one-step experiment name}"
shift
case "$method" in imf_boundary|imf_auxiliary|meanflow|direct_action|meanflow_diagonal_fix|direct_action_mae|direct_action_learnable_mae|direct_action_learnable_mse) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
source "$repo/scripts/libero/topology.sh" "$@"
export VPP2_TRAIN_MODULE=vpp2.meanflow.cli
export VPP2_CODE_REVISION="$(git rev-parse HEAD)"
export VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1 OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RUN_NAME="${RUN_NAME:-libero-${method}-${TRAIN_JOB_ID:-manual}-round${RUNNING_ROUND:-0}}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
launch=(bash "$repo/scripts/train.sh" --benchmark libero --config "$repo/configs/libero/train_${method}.yaml"
 "model.model_id=$repo/weights/Wan2.1-I2V-14B-480P"
 "model.dit_checkpoint_path=$repo/weights/libero/video_step010000.pt"
 "meanflow.student_init.action_checkpoint=$repo/weights/libero/action_init.pt"
 "meanflow.student_init.video_checkpoint=$repo/weights/libero/video_step010000.pt"
 "output_dir=$repo/runs/libero/$RUN_NAME" overwrite=false
 "batch_size=${BATCH_SIZE:-16}" "gradient_accumulation_steps=${GRAD_ACC:-1}" num_workers=4
 wandb.enabled=true wandb.mode=offline "wandb.project=${WANDB_PROJECT:-vpp2-libero}"
 "wandb.name=$RUN_NAME" "wandb.group=$RUN_NAME" "$@")
for argument in "$@"; do
 if [[ "$argument" == --dry-run ]]; then exec "${launch[@]}"; fi
done
for asset in "$repo/weights/libero/action_init.pt" "$repo/weights/libero/video_step010000.pt" "$repo/data/libero/dataset_stats.json"; do
 [[ -f "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 2; }
done
log_dir="$repo/runs/libero/${RUN_NAME}_logs"
mkdir -p "$log_dir"
exec 9> "$log_dir/node${NODE_RANK}.lock"
flock -n 9 || { echo "Launcher already running: $RUN_NAME / $NODE_RANK" >&2; exit 1; }
exec "${launch[@]}" > >(tee -a "$log_dir/node${NODE_RANK}.log") 2>&1
