#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo"
export RUN_NAME="${RUN_NAME:-recon-ddp-validation-${TRAIN_JOB_ID:-local}}"
exec bash scripts/libero_flow/train_reconstruction.sh --tokens 128 --action-denominator 8 \
  --flow-dt 4 --diagnostic max_steps=1 eval_every=0 log_every=1 \
  save_final_checkpoint=false wandb.enabled=false num_workers=12 flow.missing_motion=skip "$@"
