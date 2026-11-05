#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$root/scripts/libero_flow/train.sh" --tokens 64 --action-denominator 8 --flow-dt "${FLOW_DT:-4}" "$@"
