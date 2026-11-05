#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
source scripts/libero_flow/activate_env.sh
exec "$PYTHON_BIN" -m vpp2.libero_flow_meanflow.launch "$@"
