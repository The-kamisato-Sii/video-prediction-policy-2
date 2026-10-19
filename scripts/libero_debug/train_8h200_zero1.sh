#!/usr/bin/env bash
# Single-node, eight-GPU ZeRO1 recipe; global batch remains 256.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export NNODES="${NNODES:-${PET_NNODES:-1}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
[[ "$NNODES" == 1 && "$NPROC_PER_NODE" == 8 && "$NODE_RANK" == 0 ]] || {
  echo 'This recipe requires one node with eight GPUs.' >&2
  exit 2
}
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29618}}"
export BATCH_SIZE=32 GRAD_ACC=1 NUM_WORKERS=4
export VPP2_TRAIN_MODULE=vpp2.libero.cli
export VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export RUN_NAME="${RUN_NAME:-libero_action_8h200_b32_zero1}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo 'RUN_NAME must be a simple directory name (letters, digits, dot, underscore, hyphen).' >&2
  exit 2
}
launch=(bash scripts/libero_debug/train_8h200.sh
  "output_dir=$repo_root/runs/libero/$RUN_NAME"
  overwrite=false max_steps=30000 warmup_steps=500 save_every=5000
  wandb.enabled=true wandb.mode=offline
  "wandb.project=${WANDB_PROJECT:-vpp2-libero}"
  "wandb.name=${WANDB_NAME:-$RUN_NAME}" "wandb.group=$RUN_NAME"
  "$@")
for argument in "$@"; do
  if [[ "$argument" == --dry-run ]]; then
    exec "${launch[@]}"
  fi
done
"${PYTHON_BIN:-python}" -c 'import wandb; print("W&B", wandb.__version__, "offline")'
log_dir="$repo_root/runs/libero/${RUN_NAME}_logs"
mkdir -p "$log_dir"
exec 9> "$log_dir/node${NODE_RANK}.lock"
flock -n 9 || { echo "Node $NODE_RANK launcher is already running for $RUN_NAME" >&2; exit 1; }
exec "${launch[@]}" > >(tee -a "$log_dir/node${NODE_RANK}.log") 2>&1
