#!/usr/bin/env bash
set -euo pipefail
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
repo="$root/video-prediction-policy-2"
cd "$repo"
export LD_LIBRARY_PATH="$root/tools/libero-pro-glvnd/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python=/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro/bin/python
pids=()
for axis in object lan;do
 for suite in libero_spatial libero_object libero_goal libero_10;do
  "$python" -u scripts/libero_pro_extra/validate_assets.py --axis "$axis" --base "$suite" > "runs/libero_pro_extra_20261030/assets-${axis}-${suite}.log" 2>&1 &
  pids+=("$!")
 done
done
code=0
for pid in "${pids[@]}";do wait "$pid" || code=1;done
exit "$code"
