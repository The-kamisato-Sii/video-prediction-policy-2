#!/usr/bin/env bash
# Independent background converter; simulator generation keeps running as-is.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
session="libero-flow-motion-cache"
tmux_bin="$(command -v tmux || true)"
if [[ -z "$tmux_bin" ]]; then tmux_bin="$(dirname "$repo")/tools/tmux-portable/tmux"; fi
mkdir -p runs/libero_flow_motion_conversion
if "$tmux_bin" has-session -t "$session" 2>/dev/null; then
  echo "Already running tmux session: $session"
  exit 0
fi
printf -v command '%q ' bash "$repo/scripts/libero_flow/prepare_motions.sh" --watch --workers 2 "$@"
printf -v logfile '%q' "$repo/runs/libero_flow_motion_conversion/watcher.log"
"$tmux_bin" new-session -d -s "$session" "$command >>$logfile 2>&1"
echo "Started tmux session: $session"
echo "Status: $repo/runs/libero_flow_motion_conversion/status.json"
