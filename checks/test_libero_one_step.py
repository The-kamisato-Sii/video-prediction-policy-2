"""Controls, method semantics and production launcher contracts."""
import os
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
from omegaconf import OmegaConf
from vpp2.meanflow.cli import load_config
from vpp2.meanflow.model import MeanFlowLibero
from test_meanflow import tiny_model
from trainer_meanflow import build, Dataset


@pytest.mark.parametrize('method', ['meanflow', 'direct_action'])
def test_shared_recipe(method):
    base = OmegaConf.load('configs/libero/train_action.yaml')
    cfg = load_config(f'configs/libero/train_{method}.yaml')
    for key in ['learning_rate','optimizer_betas','weight_decay','max_grad_norm','max_steps',
                'warmup_steps','lr_scheduler_type','save_every','seed','mixed_precision','data']:
        assert OmegaConf.to_container(OmegaConf.create({'v':cfg[key]}),resolve=True) == OmegaConf.to_container(OmegaConf.create({'v':base[key]}),resolve=True)
    assert cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 1
    assert cfg.num_workers == 4 and cfg.train_mode == 'action_only'
    assert cfg.meanflow.student_init.format == 'baseline_init'
    assert cfg.meanflow.student_init.action_checkpoint == base.model.action_dit_pretrained_path
    assert cfg.meanflow.student_init.video_checkpoint == base.model.dit_checkpoint_path
    assert cfg.model.action_dit_config == base.model.action_dit_config


@pytest.mark.parametrize('method', ['meanflow', 'direct_action'])
@pytest.mark.parametrize('rank', ['0','1'])
def test_launcher_from_unrelated_cwd(tmp_path, method, rank):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash',str(repo/f'scripts/libero/train_{method}_16h200.sh'),'--dry-run'],cwd=tmp_path,
        env=dict(os.environ,VPP2_TRAIN_MODULE='invalid.probe',PYTHON_BIN=sys.executable,PET_NNODES='2',PET_NPROC_PER_NODE='8',PET_NODE_RANK=rank,PET_MASTER_ADDR='10.1.1.1'),
        capture_output=True,text=True,check=True)
    report=json.loads(result.stdout)
    assert report['global_batch']==256 and report['world_size']==16
    assert report['action_nfe']==1
    assert Path(report['student_init']['action_checkpoint']).is_absolute()


@pytest.mark.parametrize('kind', ['direct_action','data_meanflow'])
def test_frozen_video_live_action_and_proprio(kind):
    torch.manual_seed(42)
    model=tiny_model(MeanFlowLibero,checkpoint=True)
    model.build_inputs=types.MethodType(build,model)
    before={k:v.clone() for k,v in model.state_dict().items()}
    model.configure_meanflow(dict(objective=kind,derivative_backend='jvp',budget_steps=10,
        student_init={'format':'baseline_init'},interval={'diagonal_probability':0.0}),{})
    for k,v in before.items():
        torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    model.requires_grad_(False)
    model.mot.train()
    model.action_expert.requires_grad_(True)
    model.proprio_encoder.requires_grad_(True)
    model.video_expert.eval()
    sample=torch.utils.data.default_collate([Dataset()[0],Dataset()[0]])
    if kind=='direct_action':
        original=model.predict_action_velocity_with_cache
        def wrapped(**kwargs):
            assert kwargs['latents_action'].count_nonzero()==0
            assert (kwargs['timestep_action']==1000).all()
            return original(**kwargs)
        model.predict_action_velocity_with_cache=wrapped
    loss,metrics=model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.video_expert.parameters())
    for module in [model.action_expert,model.proprio_encoder]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())

@pytest.mark.parametrize('kind,sampler', [('direct_action','direct_action'),('data_meanflow','meanflow')])
def test_compact_checkpoint_restores_sampler(tmp_path, kind, sampler):
    from vpp2.libero.action_model import LiberoActionModel
    model=tiny_model(MeanFlowLibero)
    model.configure_meanflow(dict(objective=kind, student_init={'format':'baseline_init'}), {'student':{'source_step':0}})
    video=tmp_path/'video.pt'
    torch.save({'step':10000},video)
    path=tmp_path/'action.pt'
    model.save_action_checkpoint(path,str(video),step=2,resolved_cfg=OmegaConf.load('configs/libero/train_action.yaml'))
    restored=tiny_model(LiberoActionModel)
    assert restored.load_checkpoint(path,load_video_from_action_checkpoint=False)==2
    assert restored.action_sampler==sampler
    resumed=tiny_model(MeanFlowLibero)
    resumed.configure_meanflow(dict(objective=kind),{})
    assert resumed.load_checkpoint(path,load_video_from_action_checkpoint=False)==2
    assert int(resumed.meanflow_update_step)==2
    for k,v in model.action_expert.state_dict().items():
        torch.testing.assert_close(v,restored.action_expert.state_dict()[k],rtol=0,atol=0)
    for k,v in model.proprio_encoder.state_dict().items():
        torch.testing.assert_close(v,restored.proprio_encoder.state_dict()[k],rtol=0,atol=0)

def test_direct_inference_returns_clean_prediction_without_euler_step():
    m=tiny_model().eval()
    m.action_sampler='direct_action'
    m._encode_input_image_latents_tensor=lambda **kwargs: torch.zeros(1,16,1,2,2)
    m._encode_clip_fea_from_image=lambda x: None
    seen=[]
    def predict(**kwargs):
        assert kwargs['latents_action'].count_nonzero()==0
        assert (kwargs['timestep_action']==1000).all()
        seen.append(1)
        return torch.full_like(kwargs['latents_action'],0.37)
    m._predict_action_noise_with_cache=predict
    result=m.infer_action(None,torch.zeros(1,3,16,16),4,context=torch.zeros(1,3,16),
        context_mask=torch.ones(1,3,dtype=torch.bool),num_inference_steps=1,seed=1)
    assert len(seen)==1
    torch.testing.assert_close(result['action'],torch.full((4,3),0.37),rtol=0,atol=0)

def test_meanflow_padding_reduction_matches_baseline_empty_sample_weight():
    from vpp2.meanflow.objectives import data_loss
    clean=torch.zeros(2,4,3)
    pred=torch.ones_like(clean,requires_grad=True)
    pad=torch.tensor([[False]*4,[True]*4])
    def unused_derivative(*args):
        raise AssertionError('Diagonal samples require no JVP')
    loss,_=data_loss(lambda *args: pred,unused_derivative,clean,clean,
        torch.zeros(2),torch.zeros(2),pad,baseline_reduction=True)
    assert loss.item()==0.5  # Baseline averages over both samples, including the zero-loss padded row.
    loss.backward()
    assert pred.grad[1].count_nonzero()==0
