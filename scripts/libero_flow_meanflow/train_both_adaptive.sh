#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$repo/scripts/libero_flow_meanflow/train.sh" --variant adaptive --objective latent_meanflow \
  --tokens 32 --action-denominator 8 --flow-dt 4 \
  flow.motion_source=simulator_gt flow.flow_episode_scope=all \
  flow.available_manifest="$repo/runs/libero_flow_submit_20261025/flow_manifest.json" \
  flow.reconstruction_loss=mse flow.scene_flow_loss_weight=100 "$@"
