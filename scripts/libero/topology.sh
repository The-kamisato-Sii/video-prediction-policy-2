#!/usr/bin/env bash
# Source before launch. PET_* describes the allocation; otherwise count visible GPUs.
export NNODES="${NNODES:-${PET_NNODES:-1}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-}}"
if [[ -z "$NPROC_PER_NODE" ]]; then
 NPROC_PER_NODE="$("${PYTHON_BIN:-python}" -c 'import torch; print(torch.cuda.device_count())')"
 export NPROC_PER_NODE
fi
for count in "$NNODES" "$NPROC_PER_NODE"; do
 [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Node/GPU counts must be positive integers; no visible GPU?' >&2; exit 2; }
done
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${RANK:-}}}"
if [[ -z "$NODE_RANK" && "$NNODES" == 1 ]]; then export NODE_RANK=0; fi
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || { echo 'Invalid or missing NODE_RANK' >&2; exit 2; }
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
if [[ "$NNODES" == 1 ]]; then export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; fi
[[ -n "$MASTER_ADDR" ]] || { echo 'Missing master address for distributed training' >&2; exit 2; }
if (( NNODES > 1 )); then
 case "$MASTER_ADDR" in localhost|127.*|0.0.0.0|::1) echo 'Master address must reach the other node' >&2; exit 2 ;; esac
fi
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29616}}"
export WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"
(( WORLD_SIZE <= 256 && 256 % WORLD_SIZE == 0 )) || { echo "Global batch 256 is not divisible by $WORLD_SIZE GPUs with accumulation 1" >&2; exit 2; }
export GLOBAL_BATCH_SIZE=256 BATCH_SIZE="$((256 / WORLD_SIZE))" GRAD_ACC=1 NUM_WORKERS=4
for argument in "$@"; do
 case "$argument" in
 batch_size=*|gradient_accumulation_steps=*|--allow-batch-change) echo 'Batch is automatic: global=256, accumulation=1; remove batch overrides' >&2; exit 2 ;;
 esac
done
