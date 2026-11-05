#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
[[ "$(hostname)" == *e4599d9ce4b2* ]] || exit 2
cd "$repo"
pro_env=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro
export PATH="$pro_env/bin:$root/tools/tmux-portable:$PATH"
export PYTHONPATH="$repo/src" LD_LIBRARY_PATH="$root/tools/libero-pro-glvnd/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VPP2_TORCH_LOAD_MMAP=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK CUDA_VISIBLE_DEVICES
out="$repo/runs/libero_flow_eval_20261030/flow64-a1of2"
if ! tmux has-session -t flow64-half-public800 2>/dev/null;then
 tmux new-session -d -s flow64-half-public800 "$pro_env/bin/python -u $repo/scripts/libero_pro_extra/backfill_public800_20261101.py >> $out/backfill-monitor.log 2>&1"
fi
