#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
suite=${1:?suite}
case "$suite" in spatial) expected=e4599d9ce4b2;; object) expected=63c34d338f4c;; goal) expected=bf25628f9788;; long) expected=5760f5c610a5;; *) exit 2;; esac
[[ $(hostname) == *"$expected"* ]] || { echo 'Wrong suite/node assignment';exit 2; }
[[ -f "$repo/runs/libero_pseudo_flow_4rc_20261102/preview-validation.json" ]] || { echo 'Previews must be validated first';exit 2; }
cd "$repo"
source scripts/libero_pseudo_flow/activate.sh
"$PYTHON_BIN" -c 'import json;assert json.load(open("runs/libero_pseudo_flow_4rc_20261102/preview-validation.json"))["passed"]'
for gpu in 0 1;do
 session="4rc-$suite-gpu$gpu"
 if ! "$root/tools/tmux-portable/tmux" has-session -t "$session" 2>/dev/null;then
  "$root/tools/tmux-portable/tmux" new-session -d -s "$session" "bash $repo/scripts/libero_pseudo_flow/worker.sh $suite $gpu"
 fi
done
"$root/tools/tmux-portable/tmux" list-sessions
