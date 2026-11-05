#!/usr/bin/env bash
# Source this file. Do not install replay dependencies into a training environment.
vpp2_motion_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$vpp2_motion_repo/scripts/libero/activate_pro_env.sh"
vpp2_motion_runtime="${VPP2_MOTION_RUNTIME:-$vpp2_motion_repo/data/libero_motion_runtime/robosuite140}"
[[ -f "$vpp2_motion_runtime/robosuite/__init__.py" ]] || {
  echo 'Run bash scripts/libero/setup_motion_runtime.sh first.' >&2
  return 1
}
export PYTHONPATH="$vpp2_motion_runtime:$vpp2_motion_repo/third_party/LIBERO-PRO-assets:$vpp2_motion_repo/src"
export LIBERO_CONFIG_PATH="$vpp2_motion_repo/data/libero_motion_runtime/config"
unset LIBERO_TYPE
"$PYTHON_BIN" - "$vpp2_motion_repo" "$LIBERO_CONFIG_PATH" <<'PY'
from pathlib import Path
import sys, yaml
repo, config = map(Path, sys.argv[1:])
core = repo / 'third_party/LIBERO-PRO-assets/libero/libero'
paths = dict(benchmark_root=str(core), bddl_files=str(core / 'bddl_files'),
             init_states=str(core / 'init_files'), assets=str(core / 'assets'),
             datasets=str(repo / 'data/libero'))
config.mkdir(parents=True, exist_ok=True)
dest = config / 'config.yaml'
if dest.exists() and yaml.safe_load(dest.read_text()) != paths:
    raise ValueError('Motion runtime config belongs to another checkout')
if not dest.exists(): dest.write_text(yaml.safe_dump(paths))
PY
unset vpp2_motion_repo vpp2_motion_runtime
