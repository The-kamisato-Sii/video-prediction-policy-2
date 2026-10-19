#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "${NNODES:-${PET_NNODES:-2}}" == 2 && "${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}" == 8 ]] || { echo 'Requires 2 nodes x 8 GPUs' >&2; exit 2; }
export BATCH_SIZE=16 GRAD_ACC=1 VPP2_TRAIN_MODULE=vpp2.meanflow.cli
exec bash "$repo/scripts/libero/train_one_step.sh" meanflow "$@"
