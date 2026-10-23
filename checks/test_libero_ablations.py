"""Sampler distribution, controlled ablations, persistence and launcher regression."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from vpp2.meanflow.cli import load_config
from vpp2.meanflow.experiment_config import validate_experiment_resume
from vpp2.meanflow.model import MeanFlowLibero
from vpp2.meanflow.objectives import sample_intervals, masked_per_sample, data_loss
from vpp2.libero.action_model import LiberoActionModel
from test_meanflow import tiny_model
from trainer_meanflow import build, Dataset

METHODS = ['meanflow_diagonal_fix', 'direct_action_mae',
           'direct_action_learnable_mae', 'direct_action_learnable_mse']


@pytest.mark.parametrize('shift', [1., 5.])
@pytest.mark.parametrize('probability', [0., .5, 1.])
def test_controlled_sampler_and_legacy_rng(shift, probability):
    pair = torch.tensor([[.1, .4, .8, .7], [.3, .6, .2, .5]])
    mask_draw = torch.tensor([.1, .6, .3, .8])
    independent = torch.tensor([.2, .3, .4, .6])
    shifted = shift * pair / (1 + (shift - 1) * pair)
    td = shift * independent / (1 + (shift - 1) * independent)
    diagonal = mask_draw < probability
    high, low = shifted.max(0).values, shifted.min(0).values
    for mode in ['sorted_max', 'independent']:
        with patch('torch.rand', side_effect=[pair, mask_draw, independent]) as rng:
            t, r = sample_intervals(4, 'cpu', shift, probability, mode)
            assert rng.call_count == (2 if mode == 'sorted_max' else 3)
        torch.testing.assert_close(t, high if mode == 'sorted_max' else torch.where(diagonal, td, high))
        torch.testing.assert_close(r, torch.where(diagonal, high if mode == 'sorted_max' else td, low))
        assert ((0 <= r) & (r <= t) & (t <= 1)).all()
    torch.manual_seed(123)
    u = torch.rand(2, 100); s = shift * u / (1 + (shift - 1) * u)
    t, r = s.max(0).values, s.min(0).values
    expected = t, torch.where(torch.rand(100) < probability, t, r)
    next_old = torch.rand(10)
    torch.manual_seed(123)
    actual = sample_intervals(100, 'cpu', shift, probability)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    assert torch.equal(next_old, torch.rand(10))


@pytest.mark.parametrize('shift', [1., 5.])
def test_diagonal_cdf(shift):
    for mode, power in [('sorted_max', 2), ('independent', 1)]:
        torch.manual_seed(42)
        t, r = sample_intervals(100000, 'cpu', shift, 1, mode)
        assert torch.equal(t, r)
        for x in [.2, .5, .8, .9]:
            expected = (x / (shift - (shift - 1) * x)) ** power
            assert abs(float((t <= x).float().mean()) - expected) < .007
    torch.manual_seed(42); old = sample_intervals(1000, 'cpu', shift, 0)
    torch.manual_seed(42); new = sample_intervals(1000, 'cpu', shift, 0, 'independent')
    assert all(torch.equal(x, y) for x, y in zip(old, new))


@pytest.mark.parametrize('kwargs', [dict(shift=float('nan')), dict(shift=float('inf')), dict(shift=0),
    dict(diagonal_probability=float('nan')), dict(diagonal_probability=1.1), dict(diagonal_time_sampling='typo')])
def test_sampler_invalid(kwargs):
    with pytest.raises(ValueError):
        sample_intervals(4, 'cpu', **dict(dict(shift=5), **kwargs))


@pytest.mark.parametrize('mode', ['sorted_max', 'independent'])
def test_diagonal_adaptive_is_masked_fm_without_jvp(mode):
    torch.manual_seed(42)
    clean, noise = torch.randn(3, 4, 2), torch.randn(3, 4, 2)
    pad = torch.tensor([[False]*4, [False, False, True, True], [True]*4])
    t, r = sample_intervals(3, 'cpu', 5, 1, mode)
    parameter = torch.nn.Parameter(torch.tensor(.7))
    def predict(z, r, t): return parameter * z
    def derivative(*args): raise AssertionError('Diagonal must skip JVP')
    loss, stats = data_loss(predict, derivative, clean, noise, r, t, pad,
        baseline_reduction=True, loss_weighting='adaptive')
    raw, valid = masked_per_sample(predict(clean + t[:,None,None] * (noise-clean), r, t), noise-clean, pad)
    expected = torch.where(valid, raw * (raw.detach() + .001).pow(-.5), 0).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward(); assert parameter.grad.abs() > 0
    assert stats['mf/diagonal_samples'] == 2 and stats['mf/offdiagonal_samples'] == 0
    assert sum(stats[f'mf/t_diagonal_bin{i}_count'] for i in range(5)) == 2
    assert stats['mf/correction_rms_offdiagonal'] == 0


def test_masked_mae_and_padding_gradient():
    pred = torch.tensor([[[1., -3.], [99., 99.]], [[99., 99.], [99.,99.]]], requires_grad=True)
    pad = torch.tensor([[False, True], [True, True]])
    mae, valid = masked_per_sample(pred, torch.zeros_like(pred), pad, loss_type='mae')
    mse, _ = masked_per_sample(pred, torch.zeros_like(pred), pad)
    torch.testing.assert_close(mae, torch.tensor([2.,0.])); torch.testing.assert_close(mse, torch.tensor([5.,0.]))
    mae.mean().backward()
    torch.testing.assert_close(pred.grad[0,0], torch.tensor([.25,-.25]))
    assert pred.grad[0,1].count_nonzero() == pred.grad[1].count_nonzero() == 0


@pytest.mark.parametrize('loss', ['mae', 'mse'])
def test_queries_train_save_reload_and_infer(tmp_path, loss):
    torch.manual_seed(42)
    model = tiny_model(MeanFlowLibero, checkpoint=True)
    model.build_inputs = types.MethodType(build, model)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    config = dict(objective='direct_action', student_init={'format':'baseline_init'},
        direct_action=dict(input='learnable', loss=loss, query_length=4))
    model.configure_meanflow(config, {})
    for k,v in before.items(): assert torch.equal(v, model.state_dict()[k])
    query = model.action_expert.direct_action_queries
    assert query.count_nonzero() == 0 and query.numel() == 12
    model.requires_grad_(False); model.mot.train(); model.video_expert.eval()
    model.action_expert.requires_grad_(True); model.proprio_encoder.requires_grad_(True)
    sample = torch.utils.data.default_collate([Dataset()[0], Dataset()[0]])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    value, _ = model.training_loss(sample); value.backward()
    assert torch.isfinite(query.grad).all() and query.grad.norm() > 0
    assert all(p.grad is None for p in model.video_expert.parameters())
    opt.step(); assert query.count_nonzero() > 0
    video = tmp_path/'video.pt'; torch.save({'step':10000}, video)
    path = tmp_path/'action.pt'; model.save_action_checkpoint(path, str(video), step=1)
    restored = tiny_model(LiberoActionModel)
    restored.load_checkpoint(path, load_video_from_action_checkpoint=False)
    assert torch.equal(restored.action_expert.direct_action_queries, query)
    restored._encode_input_image_latents_tensor = lambda **kw: torch.zeros(1,16,1,2,2)
    restored._encode_clip_fea_from_image = lambda x: None
    calls=[]
    def predict(**kw):
        torch.testing.assert_close(kw['latents_action'], query, rtol=0, atol=0)
        calls.append(1)
        return kw['latents_action'] + .3
    restored._predict_action_noise_with_cache=predict
    result = restored.infer_action(None, torch.zeros(1,3,16,16), 4,
        context=torch.zeros(1,3,16), context_mask=torch.ones(1,3,dtype=torch.bool), num_inference_steps=1)
    assert len(calls)==1
    torch.testing.assert_close(result['action'], (query+.3).squeeze(0))
    resume=tiny_model(MeanFlowLibero); resume.configure_meanflow(config,{})
    assert resume.load_checkpoint(path,load_video_from_action_checkpoint=False)==1
    changed=copy.deepcopy(config);changed['direct_action']['loss']='mse' if loss=='mae' else 'mae'
    resume.configure_meanflow(changed,{})
    with pytest.raises(ValueError,match='differ'):resume.load_checkpoint(path,load_video_from_action_checkpoint=False)
    with pytest.raises(ValueError,match='horizon'):
        restored.action_expert.direct_action_input(torch.zeros(1,5,3))


@pytest.mark.parametrize('method', METHODS)
def test_config_controls(method):
    base = load_config('configs/libero/train_'+('meanflow' if method.startswith('meanflow') else 'direct_action')+'.yaml')
    cfg = load_config(f'configs/libero/train_{method}.yaml')
    for key in ['learning_rate','optimizer_betas','weight_decay','max_grad_norm','max_steps','warmup_steps',
                'lr_scheduler_type','save_every','seed','mixed_precision','data','num_workers','batch_size']:
        assert cfg[key] == base[key]
    assert cfg.model.action_dit_config == base.model.action_dit_config
    assert cfg.meanflow.student_init == base.meanflow.student_init
    if method.startswith('meanflow'):
        assert cfg.meanflow.interval.diagonal_time_sampling == 'independent'
        assert cfg.meanflow.interval.train_shift == 5
        assert cfg.meanflow.loss_weighting == 'adaptive'
    else:
        assert cfg.meanflow.direct_action.loss == method.split('_')[-1]
        assert cfg.meanflow.direct_action.input == ('learnable' if 'learnable' in method else 'zero')


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('nodes,rank', [('1','0'), ('2','0'), ('2','1')])
def test_launcher_any_cwd(tmp_path, method, nodes, rank):
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ, PYTHON_BIN=sys.executable, PET_NNODES=nodes, PET_NPROC_PER_NODE='8',
             PET_NODE_RANK=rank, PET_MASTER_ADDR='10.1.1.1')
    for key in ['NNODES','NPROC_PER_NODE','NODE_RANK','BATCH_SIZE','WORLD_SIZE']:
        env.pop(key,None)
    run=subprocess.run(['bash',str(repo/f'scripts/libero/train_{method}.sh'),'--dry-run'],
        cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    result=json.loads(run.stdout)
    assert result['global_batch']==256 and result['micro_batch']==256//(8*int(nodes))
    assert result['gradient_accumulation_steps']==1 and result['num_workers']==4


def test_unsupported_and_misspelled_sampler_configs():
    for override in ['meanflow.interval.diagonal_time_sampling=typo','meanflow.interval.source=typo']:
        with pytest.raises(ValueError):load_config('configs/libero/train_meanflow.yaml',[override])
    for method in ['direct_action']:
        with pytest.raises(ValueError,match='requires LIBERO'):
            load_config(f'configs/libero/train_{method}.yaml',['meanflow.interval.diagonal_time_sampling=independent'])
    for benchmark,objective in [('robodojo','data_meanflow'),('libero','kmf'),('libero','teacher_trajectory')]:
        from vpp2.meanflow.experiment_config import validate_interval_objective
        with pytest.raises(ValueError):
            validate_interval_objective(dict(objective=objective,interval=dict(diagonal_time_sampling='independent')),benchmark)


def test_resume_sampler_contract_both_paths(tmp_path):
    from vpp2.trainer import JointTrainer
    current=dict(objective='data_meanflow',interval=dict(source='shifted_uniform',diagonal_probability=.5,
        diagonal_time_sampling='independent',train_shift=5.))
    m=tiny_model(MeanFlowLibero);m.configure_meanflow(current,{})
    video=tmp_path/'video.pt';torch.save({'step':10000},video)
    path=tmp_path/'action.pt';m.save_action_checkpoint(path,str(video),step=2)
    loaded=[]
    trainer=types.SimpleNamespace(model=m,accelerator=types.SimpleNamespace(unwrap_model=lambda m:m,
        load_state=lambda **kw:loaded.append(kw)),global_step=2,epoch=0,batch_in_epoch=2)
    JointTrainer._save_trainer_state(trainer,str(tmp_path))
    for key,value in [('source','invalid'),('diagonal_probability',.25),('diagonal_time_sampling','sorted_max'),('train_shift',1.)]:
        changed=copy.deepcopy(current);changed['interval'][key]=value
        m.meanflow_config=changed
        with pytest.raises(ValueError,match='differ'):
            m.load_checkpoint(path,load_video_from_action_checkpoint=False)
        with pytest.raises(ValueError,match='differ'):
            JointTrainer.load_training_state(trainer,str(tmp_path))
        assert not loaded
    historical=dict(objective='data_meanflow',interval=dict(diagonal_probability=.5))
    old=copy.deepcopy(current);old['interval']['diagonal_time_sampling']='sorted_max'
    validate_experiment_resume(old,historical,resolved_config={'model':{'action_scheduler':{'train_shift':5}}})
    with pytest.raises(ValueError,match='Cannot verify'):
        validate_experiment_resume(old,historical)


def test_full_optimizer_query_round_trip(tmp_path):
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True)
    model = tiny_model(MeanFlowLibero)
    config = dict(objective='direct_action', direct_action=dict(loss='mae', input='learnable', query_length=4))
    model.configure_meanflow(config,{})
    model.requires_grad_(False)
    query = model.action_expert.direct_action_queries.requires_grad_(True)
    optimizer = torch.optim.AdamW([query], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(step+1))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    def update():
        loss=(model.action_expert.direct_action_queries-1).abs().mean()
        accelerator.backward(loss); optimizer.step(); scheduler.step(); optimizer.zero_grad()
    update(); accelerator.save_state(str(tmp_path/'state'))
    update(); expected = query.detach().clone()
    expected_moment = optimizer.state[query]['exp_avg'].clone()
    expected_lr = scheduler.get_last_lr()
    accelerator.load_state(str(tmp_path/'state'))
    update()
    torch.testing.assert_close(query, expected, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[query]['exp_avg'], expected_moment, rtol=0, atol=0)
    assert scheduler.get_last_lr() == expected_lr
    accelerator.free_memory()


def test_model_forwards_independent_sampler_contract():
    model=tiny_model(MeanFlowLibero)
    model.build_inputs=types.MethodType(build,model)
    model.configure_meanflow(dict(objective='data_meanflow',derivative_backend='jvp',budget_steps=30000,
        student_init=dict(format='baseline_init'),loss_weighting='adaptive',
        interval=dict(diagonal_probability=1.,diagonal_time_sampling='independent')),{})
    sample=torch.utils.data.default_collate([Dataset()[0]])
    with patch('vpp2.meanflow.model.sample_intervals', wraps=sample_intervals) as sampler:
        loss,_=model.training_loss(sample)
        assert torch.isfinite(loss)
        assert sampler.call_args.args[2:] == (model.train_action_scheduler.shift,1.,'independent')
