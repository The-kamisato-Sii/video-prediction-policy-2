#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
job="${1:?job}"; gpu="${2:?gpu}"
[[ "$job" =~ ^flow(32|64|128)-all__(public800|object|lan|env)$ && "$gpu" =~ ^[0-7]$ ]] || exit 2
cd "$repo"
folder="$repo/runs/libero_all_pro_eval_20261103/jobs/$job"
exec 9>"$folder/launch.lock"
flock -n 9 || exit 2
trap 'rc=$?; printf "%s\n" "$rc" > "$folder/exit.status"' EXIT
args=(); [[ ! -f "$folder/evaluation/RUN_CONFIG.json" ]] || args+=(--resume)
if [[ "$job" == *__public800 ]]; then
 python -u -m vpp2.libero_flow.evaluate --config "$folder/eval.yaml" --gpus "$gpu" "${args[@]}"
else
 python -u -m vpp2.libero_pro_extra.evaluate --config "$folder/eval.yaml" --gpu "$gpu" "${args[@]}"
fi
