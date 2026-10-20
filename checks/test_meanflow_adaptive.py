"""Adaptive weighting formula, masks, distributed reductions and resume contract."""
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from vpp2.meanflow.cli import load_config
from vpp2.meanflow.loss_config import loss_settings, validate_loss_resume
from vpp2.meanflow.objectives import data_loss, masked_per_sample


def objective(pred, pad, *, power=.5, mode='adaptive', baseline=True, diagonal=None):
    n=pred.shape[0]
    t=torch.ones(n) * .5
    r=t if diagonal is None else torch.where(diagonal,t,torch.zeros_like(t))
    zeros=torch.zeros_like(pred)
    return data_loss(lambda *args:pred,lambda z,r,t,v,rows:torch.zeros_like(z[rows]),
        zeros,zeros,r,t,pad,baseline_reduction=baseline,
        loss_weighting=mode,adaptive_power=power,adaptive_c=.001)


@pytest.mark.parametrize('baseline',[True,False])
@pytest.mark.parametrize('all_pad',[True,False])
def test_power_zero_uniform_value_and_gradient(baseline,all_pad):
    pred=torch.arange(12,dtype=torch.float32).reshape(2,3,2).requires_grad_()
    pad=torch.tensor([[False,True,False],[True,True,True]])
    if all_pad:pad[:]=True
    a,_=objective(pred,pad,power=0,baseline=baseline)
    b,_=objective(pred,pad,mode='uniform',baseline=baseline)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga,=torch.autograd.grad(a,pred,retain_graph=True);gb,=torch.autograd.grad(b,pred)
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    assert ga[pad].count_nonzero()==0


def test_official_detached_weight_formula_and_mask_invariance():
    pred=torch.tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]],requires_grad=True)
    pad=torch.tensor([[False,True],[False,False]])
    loss,metrics=objective(pred,pad,diagonal=torch.tensor([True,False]))
    e,_=masked_per_sample(pred,torch.zeros_like(pred),pad)
    # Official denominator formulation, adapted to the documented valid-element mean.
    reference=(e/(e.detach()+.001).pow(.5)).mean()
    torch.testing.assert_close(loss,reference)
    loss.backward()
    count=(~pad).sum(1)*2
    expected=2*pred.detach()/count[:,None,None]/2/(e.detach()+.001).sqrt()[:,None,None]
    expected[pad]=0
    torch.testing.assert_close(pred.grad,expected)
    assert pred.grad.norm()>0
    changed=pred.detach().clone();changed[pad]=10000;changed.requires_grad_()
    other,_=objective(changed,pad);other.backward()
    torch.testing.assert_close(loss,other);torch.testing.assert_close(pred.grad,changed.grad)
    assert metrics['mf/diagonal_samples']==1 and metrics['mf/offdiagonal_samples']==1
    assert metrics['mf/valid_samples']==2
    assert loss.dtype==torch.float32


def test_defaults_and_invalid_configs(tmp_path):
    cfg=load_config('configs/libero/train_meanflow.yaml')
    assert loss_settings(cfg.meanflow)==dict(loss_weighting='adaptive',adaptive_weighting=dict(power=.5,c=.001))
    assert load_config('configs/libero/train_direct_action.yaml').meanflow.loss_weighting=='uniform'
    for override in ['meanflow.loss_weighting=adptive','meanflow.adaptive_weighting.power=-.1',
        'meanflow.adaptive_weighting.power=1.1','meanflow.adaptive_weighting.power=nan',
        'meanflow.adaptive_weighting.c=0','meanflow.adaptive_weighting.c=inf']:
        with pytest.raises(ValueError):load_config('configs/libero/train_meanflow.yaml',[override])
    with pytest.raises(ValueError):load_config('configs/libero/train_direct_action.yaml',['meanflow.loss_weighting=adaptive'])
    legacy=OmegaConf.to_container(cfg,resolve=True)
    legacy['meanflow'].pop('loss_weighting');legacy['meanflow'].pop('adaptive_weighting')
    path=tmp_path/'legacy.yaml';OmegaConf.save(OmegaConf.create(legacy),path)
    assert load_config(path).meanflow.loss_weighting=='uniform'
    validate_loss_resume({}, {})
    validate_loss_resume(cfg.meanflow,cfg.meanflow)
    for saved in [{},{'loss_weighting':'adaptive','adaptive_weighting':{'power':1,'c':.001}},
                  {'loss_weighting':'adaptive','adaptive_weighting':{'power':.5,'c':.01}}]:
        with pytest.raises(ValueError,match='differ'):validate_loss_resume(cfg.meanflow,saved)


def _distributed_worker(rank,init_file,out_dir):
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2,timeout=timedelta(seconds=45))
    try:
        for baseline in [True,False]:
            for empty_rank in [True,False]:
                # Rank zero has no off-diagonal examples; optionally no valid examples at all.
                pad=torch.zeros(2,2,dtype=torch.bool)
                if rank==0 and empty_rank:pad[:]=True
                p=torch.tensor(2.,requires_grad=True)
                pred=p.expand(2,2,1)*(rank+1)
                diag=torch.ones(2,dtype=torch.bool) if rank==0 else torch.zeros(2,dtype=torch.bool)
                loss,metrics=objective(pred,pad,baseline=baseline,diagonal=diag)
                loss.backward()
                grad=p.grad.clone();dist.all_reduce(grad);grad/=2
                stats={k:float(v) for k,v in metrics.items() if k.startswith('mf/')}
                Path(out_dir,f'{baseline}-{empty_rank}-{rank}.json').write_text(json.dumps(dict(grad=float(grad),metrics=stats)))
        # Rank-zero-only evaluation must not enter training collectives.
        if rank==0:
            zeros=torch.zeros(2,2,1)
            data_loss(lambda *args:zeros,lambda *args:zeros,zeros,zeros,
                      torch.zeros(2),torch.zeros(2),loss_weighting='adaptive',distributed=False)
            import types
            from test_meanflow import tiny_model
            from trainer_meanflow import build, Dataset
            from vpp2.meanflow.model import MeanFlowLibero
            model=tiny_model(MeanFlowLibero)
            model.configure_meanflow(dict(objective='data_meanflow',loss_weighting='adaptive',
                derivative_backend='jvp',budget_steps=30000,student_init={'format':'baseline_init'},
                interval={'diagonal_probability':.5}),{})
            model.build_inputs=types.MethodType(build,model)
            model.eval()
            with torch.no_grad():
                loss,_=model.training_loss(torch.utils.data.default_collate([Dataset()[0]]))
            assert torch.isfinite(loss)

        dist.barrier()
    finally:dist.destroy_process_group()


def test_two_rank_empty_groups_and_gradient_reference(tmp_path):
    mp.spawn(_distributed_worker,args=(str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    for baseline in [True,False]:
        for empty_rank in [True,False]:
            p=torch.tensor(2.,requires_grad=True)
            pred=p.expand(4,2,1)*torch.tensor([1.,1.,2.,2.])[:,None,None]
            pad=torch.zeros(4,2,dtype=torch.bool)
            if empty_rank:pad[:2]=True
            loss,metrics=objective(pred,pad,baseline=baseline,diagonal=torch.tensor([True,True,False,False]))
            loss.backward()
            for rank in range(2):
                row=json.loads((tmp_path/f'{baseline}-{empty_rank}-{rank}.json').read_text())
                assert row['grad']==pytest.approx(float(p.grad),rel=1e-6)
                for k,v in row['metrics'].items():assert v==pytest.approx(float(metrics[k]),rel=1e-6)


def test_adaptive_compact_metadata_and_reject_uniform_resume(tmp_path):
    from test_meanflow import tiny_model
    from vpp2.meanflow.model import MeanFlowLibero
    config=dict(objective='data_meanflow',loss_weighting='adaptive',adaptive_weighting=dict(power=.5,c=.001))
    model=tiny_model(MeanFlowLibero);model.configure_meanflow(config,{})
    video=tmp_path/'video.pt';torch.save({'step':10000},video)
    path=tmp_path/'action.pt'
    model.save_action_checkpoint(path,str(video),step=2,resolved_cfg={'meanflow':config})
    payload=torch.load(path,weights_only=True)
    assert loss_settings(payload['meanflow']['config'])==loss_settings(config)
    restored=tiny_model(MeanFlowLibero);restored.configure_meanflow(config,{})
    assert restored.load_checkpoint(path,load_video_from_action_checkpoint=False)==2
    restored.configure_meanflow(dict(objective='data_meanflow'),{})
    with pytest.raises(ValueError,match='differ'):restored.load_checkpoint(path,load_video_from_action_checkpoint=False)


def test_full_state_loss_metadata_and_guard_before_loading(tmp_path):
    from types import SimpleNamespace
    from vpp2.trainer import JointTrainer
    cfg=dict(objective='data_meanflow',loss_weighting='adaptive',adaptive_weighting=dict(power=.5,c=.001))
    model=SimpleNamespace(meanflow_config=cfg)
    loaded=[]
    accelerator=SimpleNamespace(unwrap_model=lambda m:m,load_state=lambda **kwargs:loaded.append(kwargs))
    trainer=SimpleNamespace(model=model,accelerator=accelerator,global_step=4,epoch=0,batch_in_epoch=4)
    JointTrainer._save_trainer_state(trainer,str(tmp_path))
    saved=json.loads((tmp_path/'trainer_state.json').read_text())
    assert saved['meanflow_config']==cfg
    model.meanflow_config={'objective':'data_meanflow'}
    with pytest.raises(ValueError,match='differ'):
        JointTrainer.load_training_state(trainer,str(tmp_path))
    assert loaded==[]
