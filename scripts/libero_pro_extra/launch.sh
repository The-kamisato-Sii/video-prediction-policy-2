#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
node="${1:?h200-a or h200-b}"
case "$node" in h200-a) expected=e4599d9ce4b2;; h200-b) expected=63c34d338f4c;; *) exit 2;; esac
[[ "$(hostname)" == *"$expected"* ]] || exit 2
python=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro/bin/python
tmux="$root/tools/tmux-portable/tmux"
out="$repo/runs/libero_pro_extra_20261030"
mkdir -p "$out/nodes/$node"
if ! "$tmux" has-session -t flow2action-extra-pro 2>/dev/null;then
 "$tmux" new-session -d -s flow2action-extra-pro "$python -u $repo/scripts/libero_pro_extra/distributed_scheduler.py --node $node >> $out/nodes/$node/queue.log 2>&1"
fi
"$tmux" list-sessions
