#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
while [[ ! -f weights/4RC/model.safetensors ]];do sleep 15;done
[[ $(stat -Lc %s weights/4RC/model.safetensors) == 6080387740 ]]
exec bash scripts/libero_pseudo_flow/preview.sh "$@"
