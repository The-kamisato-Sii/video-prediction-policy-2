#!/usr/bin/env bash
# Original RoboDojo FM baseline with the reference global batch on eight H200s.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export REQUIRE_RDMA="${REQUIRE_RDMA:-0}" VPP2_TRAIN_MODULE=vpp2.cli
export VPP2_ACCELERATE_CONFIG="${VPP2_ACCELERATE_CONFIG:-configs/robodojo/accelerate_zero2.yaml}"
exec bash "$repo_dir/scripts/train.sh" --config configs/robodojo/train_8h200.yaml "$@"
