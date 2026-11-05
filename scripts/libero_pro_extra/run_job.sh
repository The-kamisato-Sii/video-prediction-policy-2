#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
out="$repo/runs/libero_pro_extra_20261030"
job="${1:?job}";gpu="${2:?gpu}"
[[ "$job" =~ ^[a-zA-Z0-9_-]+__(object|lan|env)$ && "$gpu" =~ ^[01]$ ]] || exit 2
cd "$repo"
folder="$out/jobs/$job"
exec 9>"$folder/launch.lock"
flock -n 9 || exit 2
trap 'rc=$?; printf "%s\n" "$rc" > "$folder/exit.status"' EXIT
pro_env=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro
export PATH="$pro_env/bin:$root/tools/tmux-portable:$PATH"
export PYTHONPATH="$repo/src"
export LD_LIBRARY_PATH="$root/tools/libero-pro-glvnd/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VPP2_TORCH_LOAD_MMAP=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK CUDA_VISIBLE_DEVICES
args=();if [[ -f "$folder/evaluation/RUN_CONFIG.json" ]];then args+=(--resume);fi
"$pro_env/bin/python" -u -m vpp2.libero_pro_extra.evaluate --config "$folder/eval.yaml" --gpu "$gpu" "${args[@]}"
