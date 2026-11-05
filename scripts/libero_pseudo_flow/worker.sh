#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero_pseudo_flow/activate.sh
suite=${1:?suite};gpu=${2:?gpu}
export CUDA_VISIBLE_DEVICES="$gpu"
out="runs/libero_pseudo_flow_4rc_20261102/production/$suite"
mkdir -p "$out"
exec 9>"$out/gpu${gpu}.lock"
flock -n 9 || exit 0
exec "$PYTHON_BIN" -m vpp2.libero_pseudo_flow.extract --suite "$suite" --checkpoint weights/4RC/model.safetensors --resolution 504 --block 16 --pair-batch 4 --status "$out/gpu${gpu}.json" --code-revision "$(git -C ../tools/4RC rev-parse HEAD)" --parity >> "$out/gpu${gpu}.log" 2>&1
