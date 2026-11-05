#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
job="${1:?job}";gpu="${2:?gpu}"
[[ "$job" =~ ^4rc-flow32-a1of2__(public800|object|lan|env)_part[0-3]$ && "$gpu" =~ ^[0-7]$ ]] || exit 2
cd "$repo"
folder="$repo/runs/libero_4rc32_a1of2_pro_eval_20261104/jobs/$job"
exec 9>"$folder/launch.lock"
flock -n 9 || exit 2
trap 'rc=$?; printf "%s\n" "$rc" > "$folder/exit.status"' EXIT
args=(); [[ ! -f "$folder/evaluation/RUN_CONFIG.json" ]] || args+=(--resume)
python -u scripts/libero_4rc_pro_eval/evaluate_shard.py --config "$folder/eval.yaml" --gpu "$gpu" "${args[@]}"
