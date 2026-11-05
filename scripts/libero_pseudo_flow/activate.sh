#!/usr/bin/env bash
pseudo_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON_BIN="$pseudo_repo/runs/libero_pseudo_flow_4rc_20261102/venv/bin/python"
export PYTHONPATH="$pseudo_repo/src:$(dirname "$pseudo_repo")/tools/4RC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
unset pseudo_repo
