#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
python -u -m accelerate.commands.launch \
  --config_file configs/robodojo/accelerate_zero2.yaml \
  checks/profile_frozen_video.py "$@"
