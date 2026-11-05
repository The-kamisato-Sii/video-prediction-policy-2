#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
cd "$repo"
out="$repo/runs/libero_recon128_pro_eval_20261104"
pro_env=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro
export PATH="$pro_env/bin:$root/tools/tmux-portable:$PATH"
export PYTHONPATH="$repo/runs/libero_pro_extra_20261030/runtime:$repo/src"
export LD_LIBRARY_PATH="$root/tools/libero-pro-glvnd/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VPP2_TORCH_LOAD_MMAP=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK CUDA_VISIBLE_DEVICES
exec 9>"$out/node-launcher.lock"
flock -n 9 || exit 2
python -u scripts/libero_recon128_pro_eval/validate.py > "$out/validation.log" 2>&1
if ! tmux has-session -t libero-all-pro-gpu-guard 2>/dev/null; then
 tmux new-session -d -s libero-all-pro-gpu-guard "python -u $repo/runs/libero_all_pro_eval_20261103/gpu_idle_guard.py >> $repo/runs/libero_all_pro_eval_20261103/gpu-guard.log 2>&1"
fi
if ! tmux has-session -t libero-recon128-pro-eval 2>/dev/null; then
 tmux new-session -d -s libero-recon128-pro-eval "python -u $repo/scripts/libero_recon128_pro_eval/controller.py >> $out/queue.log 2>&1"
fi
tmux list-sessions
