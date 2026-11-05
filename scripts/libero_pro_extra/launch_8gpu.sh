#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
node="${1:?h200-c or h200-d}"
case "$node" in h200-c) expected=bf25628f9788;; h200-d) expected=5760f5c610a5;; *) exit 2;; esac
[[ "$(hostname)" == *"$expected"* ]] || exit 2
cd "$repo"
pro_env=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro
python="$pro_env/bin/python"
tmux="$root/tools/tmux-portable/tmux"
out="$repo/runs/libero_pro_extra_20261030"
mkdir -p "$out/nodes/$node"
export PATH="$pro_env/bin:$root/tools/tmux-portable:$PATH"
export PYTHONPATH="$out/runtime:$repo/src"
export LD_LIBRARY_PATH="$root/tools/libero-pro-glvnd/lib:${LD_LIBRARY_PATH:-}"
export LIBERO_CONFIG_PATH="$out/runtime_config" LIBERO_TYPE=pro MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK CUDA_VISIBLE_DEVICES
if ! "$tmux" has-session -t flow2action-gpu-guard 2>/dev/null;then
 "$tmux" new-session -d -s flow2action-gpu-guard "$python -u $out/nodes/$node/gpu_idle_guard.py >> $out/nodes/$node/gpu-guard.log 2>&1"
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$out/nodes/$node/gpu-initial.csv"
"$python" -u scripts/libero_pro_extra/preflight_new_node.py --node "$node" > "$out/nodes/$node/preflight.log" 2>&1
if ! "$tmux" has-session -t flow2action-extra-pro 2>/dev/null;then
 "$tmux" new-session -d -s flow2action-extra-pro "$python -u $repo/scripts/libero_pro_extra/distributed_scheduler_8gpu.py --node $node >> $out/nodes/$node/queue.log 2>&1"
fi
"$tmux" list-sessions
