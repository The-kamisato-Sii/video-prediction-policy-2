#!/usr/bin/env bash
set -euo pipefail
suite="${1:?suite required}"
case "$suite" in object|goal|spatial|10) ;; *) exit 2;; esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="$(dirname "$repo")"
cd "$repo"
source scripts/libero/activate_motion_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run="runs/libero_motion_suites_20261023_v3/$suite"
mkdir -p "$run"
if [[ "${2:-}" == worker ]]; then
  exec "$PYTHON_BIN" -u scripts/libero/generate_all_motions.py --suite "$suite" --run-dir "$run" >>"$run/supervisor.log" 2>&1
fi
tmux_bin="$(command -v tmux || true)"
if [[ -z "$tmux_bin" ]]; then tmux_bin="$base/tools/tmux-portable/tmux"; fi
if [[ "$suite" != object ]] && ! "$tmux_bin" has-session -t motion-gpu 2>/dev/null; then
  "$tmux_bin" new-session -d -s motion-gpu "$base/miniconda/envs/vpp2/bin/python -u $base/test_gpu.py >/dev/null 2>&1"
fi
if ! "$tmux_bin" has-session -t "motion-$suite" 2>/dev/null; then
  "$tmux_bin" new-session -d -s "motion-$suite" "bash $repo/scripts/libero/start_motion_suite.sh $suite worker"
fi
"$tmux_bin" list-sessions
