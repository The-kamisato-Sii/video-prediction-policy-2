#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$repo/scripts/libero_flow/train_reconstruction.sh" --tokens 64 --action-denominator 4 \
  --flow-dt "${FLOW_DT:-4}" "$@"
