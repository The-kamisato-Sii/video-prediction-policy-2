#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: train.sh --benchmark robodojo|libero --config configs/meanflow/...yaml [--dry-run] [key=value ...]'
  echo 'Defaults to one node / one GPU. Set accumulation to preserve the reference global batch.'
  exit 0
fi
export NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-1}" REQUIRE_RDMA="${REQUIRE_RDMA:-0}"
export VPP2_TRAIN_MODULE=vpp2.meanflow.cli
benchmark=robodojo
arguments=("$@")
for index in "${!arguments[@]}"; do
  if [[ "${arguments[index]}" == --benchmark ]]; then
    benchmark="${arguments[index+1]:-}"
  elif [[ "${arguments[index]}" == --benchmark=* ]]; then
    benchmark="${arguments[index]#--benchmark=}"
  fi
done
if [[ "$benchmark" == libero ]]; then
  default_accelerate=configs/libero/accelerate_zero2.yaml
else
  default_accelerate=configs/robodojo/accelerate.yaml
fi
export VPP2_ACCELERATE_CONFIG="${VPP2_ACCELERATE_CONFIG:-$default_accelerate}"
exec bash scripts/train.sh "$@"
