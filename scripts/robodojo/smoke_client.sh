#!/usr/bin/env bash
# One native RoboDojo episode; this is an execution probe, not the 2100-episode benchmark.
set -euo pipefail
sim_root=""
sim_python=""
host="127.0.0.1"
port="23000"
task="stack_bowls"
gpu="0"
seed="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --robodojo-root) sim_root="$2"; shift 2 ;;
    --sim-python) sim_python="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --robodojo-root PATH --sim-python PATH [--host HOST --port PORT --task TASK --gpu GPU --seed SEED]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$sim_root/scripts/robodojo.sh" && -x "$sim_python" ]] || {
  echo "Provide a simulator checkout and its Python executable." >&2; exit 2;
}
sim_root="$(cd "$sim_root" && pwd)"
sim_python="$(cd "$(dirname "$sim_python")" && pwd)/$(basename "$sim_python")"
export PATH="$(dirname "$sim_python"):$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJO_MAX_BASH_RETRIES=0
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-vpp2_smoke_$(date -u +%Y%m%d_%H%M%S)_$$}"
exec bash "$sim_root/scripts/robodojo.sh" client --task "$task" \
  --policy-name VPP2 --policy-host "$host" --policy-port "$port" \
  --env-gpu "$gpu" --seed "$seed" --eval-num 1 --ckpt "$ROBODOJO_RUN_ID"
