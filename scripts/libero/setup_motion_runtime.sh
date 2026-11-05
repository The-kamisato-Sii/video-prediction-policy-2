#!/usr/bin/env bash
# Isolated robosuite 1.4 replay runtime; reuse PRO's MuJoCo 3.3.2 dependencies.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="${VPP2_MOTION_RUNTIME:-$repo/data/libero_motion_runtime/robosuite140}"
wheel="${1:-$repo/data/libero_motion_runtime/robosuite-1.4.0-py3-none-any.whl}"
mkdir -p "$(dirname "$wheel")" "$runtime"
if [[ ! -f "$runtime/robosuite/__init__.py" ]]; then
  if [[ ! -f "$wheel" ]]; then
    curl -L --fail --retry 2 --max-time 600 \
      'https://files.pythonhosted.org/packages/2e/08/fe231064caaf2766d47818ca12fd8acf10dc4e762a33dabc6293e83bfead/robosuite-1.4.0-py3-none-any.whl' \
      -o "$wheel"
  fi
  "$repo/../miniconda/envs/vpp2-libero-pro/bin/python" - "$wheel" "$runtime" <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
wheel = Path(sys.argv[1])
expected = 'aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994'
if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
    raise ValueError('Robosuite wheel checksum mismatch')
with zipfile.ZipFile(wheel) as z:
    z.extractall(sys.argv[2])
PY
fi
echo "Motion runtime: $runtime"
