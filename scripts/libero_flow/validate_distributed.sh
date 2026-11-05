#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
export RUN_NAME="${RUN_NAME:-flow-ddp-validation-${TRAIN_JOB_ID:-local}}"
# Real full-size pretrained model, two GPUs, one effective256-sample optimizer
# update. Incomplete GT is explicitly allowed ONLY for this diagnostic.
exec bash scripts/libero_flow/train.sh --tokens 128 --action-denominator 8 \
  --flow-dt 4 --diagnostic max_steps=1 eval_every=0 log_every=1 \
  save_final_checkpoint=false wandb.enabled=false num_workers=2 \
  flow.missing_motion=skip "$@"
