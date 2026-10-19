#!/usr/bin/env bash
# Keep the 17-frame logical Video sequence; decode only the conditioning frame.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export RUN_NAME="${RUN_NAME:-libero_action_8h200_b32_zero1_first_frame}"
exec bash scripts/libero/train_8h200_zero1.sh "$@" \
  train_mode=action_only model.action_only_first_frame_fast_path=true \
  data.train.decode_only_first_video_frame=true \
  data.train.decode_sampled_video_frames=false \
  data.train.processor.num_image_obs_steps=1
