"""Allocation-derived batch and visible-device discovery contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('method', ['action', 'meanflow', 'direct_action'])
@pytest.mark.parametrize('nodes,gpus', [(1, 8), (2, 8), (4, 8)])
def test_allocation_batch(tmp_path, method, nodes, gpus):
    capture = tmp_path / 'python'
    capture.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps(dict(args=sys.argv[1:],world=os.environ["WORLD_SIZE"])))\n')
    capture.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PET_') and k not in ('NNODES','NPROC_PER_NODE','NODE_RANK','RANK','MASTER_ADDR','MASTER_PORT')}
    env.update(PYTHON_BIN=str(capture), PET_NNODES=str(nodes), PET_NPROC_PER_NODE=str(gpus), PET_NODE_RANK='0', PET_MASTER_ADDR='10.0.0.1', BATCH_SIZE='999', GRAD_ACC='99')
    x = subprocess.run(['bash',str(REPO/f'scripts/libero/train_{method}.sh'),'--dry-run'],cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    report = json.loads(x.stdout)
    assert report['world'] == str(nodes*gpus)
    assert f'batch_size={256//(nodes*gpus)}' in report['args']
    assert 'gradient_accumulation_steps=1' in report['args']


@pytest.mark.parametrize('visible', [8, 0, 3])
def test_local_discovery(tmp_path, visible):
    capture = tmp_path / 'python'
    capture.write_text(f'#!/bin/sh\nprintf "{visible}\\n"\n')
    capture.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PET_') and k not in ('NNODES','NPROC_PER_NODE','NODE_RANK','RANK','MASTER_ADDR','MASTER_PORT')}
    env['PYTHON_BIN'] = str(capture)
    x = subprocess.run(['bash','-c',f'set -euo pipefail; source "{REPO}/scripts/libero/topology.sh"; echo "$WORLD_SIZE $BATCH_SIZE $GRAD_ACC"'],env=env,text=True,capture_output=True)
    if visible == 8:
        assert x.returncode == 0 and x.stdout.strip() == '8 32 1'
    else:
        assert x.returncode == 2


def test_batch_override_rejected():
    env = dict(os.environ, NNODES='1',NPROC_PER_NODE='8',NODE_RANK='0')
    x = subprocess.run(['bash',str(REPO/'scripts/libero/train_action.sh'),'--dry-run','batch_size=1'],env=env,capture_output=True,text=True)
    assert x.returncode == 2 and 'Batch is automatic' in x.stderr
