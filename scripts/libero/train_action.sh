#!/usr/bin/env bash
# FM baseline: run once per node; batch is derived from the GPU allocation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/libero/topology.sh" "$@"
export VPP2_TRAIN_MODULE=vpp2.libero.cli
export VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export RUN_NAME="${RUN_NAME:-libero-action-${TRAIN_JOB_ID:-manual}-round${RUNNING_ROUND:-0}}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo 'RUN_NAME must be a simple directory name (letters, digits, dot, underscore, hyphen).' >&2
  exit 2
}
launch=(bash scripts/libero/train_baseline.sh
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
