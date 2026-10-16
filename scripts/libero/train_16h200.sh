#!/usr/bin/env bash
# Run once on each of two GPU nodes; each node owns eight H200 GPUs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export NNODES=2 NPROC_PER_NODE=8
# Qizhi PyTorch jobs may supply node RANK before the local launcher starts.
# Explicit NODE_RANK takes precedence, including for manual interactive launches.
export NODE_RANK="${NODE_RANK:-${RANK:-}}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || {
  echo 'Set NODE_RANK=0 on the master and NODE_RANK=1 on the worker.' >&2
  exit 2
}
: "${MASTER_ADDR:?Set MASTER_ADDR to the master node address reachable from both nodes}"
case "$MASTER_ADDR" in
  localhost|127.*|0.0.0.0|::1)
    echo 'MASTER_ADDR must be reachable from the other node, not a loopback address.' >&2
    exit 2 ;;
esac
export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29616}"
export BATCH_SIZE=16 GRAD_ACC=1 NUM_WORKERS=16
export VPP2_TRAIN_MODULE=vpp2.libero.cli
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export RUN_NAME="${RUN_NAME:-libero_action_16h200_b16}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo 'RUN_NAME must be a simple directory name (letters, digits, dot, underscore, hyphen).' >&2
  exit 2
}
launch=(bash scripts/libero/train_8h200.sh
  "output_dir=$repo_root/runs/libero/$RUN_NAME"
  overwrite=false max_steps=30000 warmup_steps=500 save_every=5000
  wandb.enabled=true wandb.mode=offline wandb.project=vpp2-libero
  "wandb.name=$RUN_NAME" "wandb.group=$RUN_NAME"
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
"${launch[@]}" 2>&1 | tee -a "$log_dir/node${NODE_RANK}.log"
