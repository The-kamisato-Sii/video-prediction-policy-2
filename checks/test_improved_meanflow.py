"""iMF equation, gradient boundaries, architecture and checkpoint contracts."""
import copy
import types
import pytest
import torch
from omegaconf import OmegaConf
from test_meanflow import tiny_model, inputs
from trainer_meanflow import Dataset, build
from vpp2.meanflow.cli import load_config
from vpp2.meanflow.model import MeanFlowLibero, prepare_condition, derivative_mode
from vpp2.libero.action_model import LiberoActionModel
from vpp2.meanflow.objectives import improved_data_loss, data_loss, directional_derivative
from vpp2.meanflow.experiment_config import validate_experiment_resume


def config(variant):
    return dict(objective='data_improved_meanflow', student_init={'format':'baseline_init'},
                derivative_backend='jvp', budget_steps=30000, loss_weighting='adaptive',
                adaptive_weighting=dict(power=.5,c=.001),
                interval=dict(diagonal_time_sampling='independent'), imf=dict(variant=variant))


@pytest.mark.parametrize('auxiliary', [False, True])
def test_equation_detachment_and_separate_adaptive(auxiliary):
    torch.manual_seed(42)
    p=torch.nn.Parameter(torch.tensor(.7)); q=torch.nn.Parameter(torch.tensor(.4))
    x=torch.randn(3,4,2); e=torch.randn_like(x)
    r=torch.tensor([.2,.3,.4]); t=torch.tensor([.8,.3,.7])
    pad=torch.tensor([[False]*4,[False,False,True,True],[True]*4])
    vc=e-x;z=x+t[:,None,None]*vc
    predict=lambda z,r,t:p*z+t[:,None,None]
    def vpred(z,t,rows):return q*z[rows]+t[rows,None,None]
    def derivative(z,r,t,direction,rows):
        assert not direction.requires_grad
        return directional_derivative(predict,z[rows],r[rows],t[rows],direction[rows])
    loss,_=improved_data_loss(predict,derivative,vpred,x,e,r,t,pad,auxiliary=auxiliary,distributed=False)
    loss.backward(); actualp=p.grad.clone(); actualq=None if q.grad is None else q.grad.clone()
    p.grad=q.grad=None
    vp=q*z+t[:,None,None]
    d=(p*vp+1).detach()
    target=(vc-(t-r)[:,None,None]*d).detach()
    from vpp2.meanflow.objectives import masked_per_sample
    eu,_=masked_per_sample(predict(z,r,t),target,pad)
    ev,_=masked_per_sample(vp,vc,pad)
    expected=(eu/(eu.detach()+.001).sqrt()).mean()
    if auxiliary:expected=expected+(ev/(ev.detach()+.001).sqrt()).mean()
    expected.backward()
    torch.testing.assert_close(loss,expected);torch.testing.assert_close(actualp,p.grad)
    if auxiliary:torch.testing.assert_close(actualq,q.grad)
    else:assert actualq is q.grad is None


def test_conditional_direction_recovers_old_meanflow():
    x=torch.randn(2,3,2); e=torch.randn_like(x);r=torch.zeros(2);t=torch.ones(2)*.7
    predict=lambda z,r,t: z*.5
    derivative=lambda z,r,t,v,rows:v[rows]*.5
    a,_=improved_data_loss(predict,derivative,lambda z,t,rows:(e-x)[rows],x,e,r,t,distributed=False)
    b,_=data_loss(predict,derivative,x,e,r,t,baseline_reduction=True,loss_weighting='adaptive',distributed=False)
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('variant',['boundary','auxiliary'])
def test_model_training_checkpoint_and_inference(tmp_path,variant):
    torch.manual_seed(42)
    m=tiny_model(MeanFlowLibero,layers=40,checkpoint=True)
    m.configure_meanflow(config('boundary'),{})
    # Capture original u weights and RNG after its normal endpoint initialization.
    before={k:v.clone() for k,v in m.action_expert.state_dict().items()}
    rng=torch.get_rng_state().clone()
    if variant=='auxiliary':m.action_expert.enable_aux_v()
    assert torch.equal(rng,torch.get_rng_state())
    assert len(m.action_expert.blocks)==40
    for k,v in before.items():assert torch.equal(v,m.action_expert.state_dict()[k])
    if variant=='auxiliary':
        for src,dst in zip(m.action_expert.blocks[-8:].parameters(),m.action_expert.aux_v_blocks.parameters()):
            assert src.data_ptr()!=dst.data_ptr();torch.testing.assert_close(src,dst,rtol=0,atol=0)
    # Fresh model avoids registering duplicate gradient hooks in normal use.
    m=tiny_model(MeanFlowLibero,layers=40,checkpoint=True);m.configure_meanflow(config(variant),{})
    m.build_inputs=types.MethodType(build,m)
    m.requires_grad_(False);m.action_expert.requires_grad_(True);m.proprio_encoder.requires_grad_(True)
    m.mot.train();m.video_expert.eval()
    sample=torch.utils.data.default_collate([Dataset()[0],Dataset()[0]])
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    loss,metrics=m.training_loss(sample);loss.backward()
    assert torch.isfinite(loss)
    for module in [m.action_expert.blocks,m.action_expert.head,m.proprio_encoder]+([m.action_expert.aux_v_blocks,m.action_expert.aux_v_head] if variant=='auxiliary' else []):
        assert any(p.grad is not None and p.grad.norm()>0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(p.grad is None for p in m.video_expert.parameters())
    opt.step();m.consume_imf_gradient_metrics()
    # Full optimizer and model state preserve auxiliary moments too.
    full=tmp_path/'full.pt';m.save_checkpoint(full,opt,step=1)
    restored=tiny_model(MeanFlowLibero,layers=40);restored.configure_meanflow(config(variant),{})
    restored.requires_grad_(False);restored.action_expert.requires_grad_(True);restored.proprio_encoder.requires_grad_(True)
    ropt=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.001)
    assert restored.load_checkpoint(full,ropt)==1
    assert len(ropt.state)==len(opt.state)>0
    for p,q in zip(opt.param_groups[0]['params'],ropt.param_groups[0]['params']):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        if p in opt.state:torch.testing.assert_close(opt.state[p]['exp_avg'],ropt.state[q]['exp_avg'],rtol=0,atol=0)
    video=tmp_path/'video.pt';torch.save({'step':10000},video)
    path=tmp_path/'action.pt';m.save_action_checkpoint(path,str(video),step=1)
    deploy=tiny_model(LiberoActionModel,layers=40)
    deploy.load_checkpoint(path,load_video_from_action_checkpoint=False)
    for k,v in m.action_expert.state_dict().items():torch.testing.assert_close(v,deploy.action_expert.state_dict()[k],rtol=0,atol=0)
    deploy.eval();deploy._encode_input_image_latents_tensor=lambda **kw:torch.zeros(1,16,1,2,2)
    deploy._encode_clip_fea_from_image=lambda x:None
    seen=[];aux=[]
    h=deploy.action_expert.action_encoder.register_forward_hook(lambda *args:seen.append(1))
    ah=deploy.action_expert.aux_v_head.register_forward_hook(lambda *args:aux.append(1)) if variant=='auxiliary' else None
    result=deploy.infer_action(None,torch.zeros(1,3,16,16),4,context=torch.zeros(1,3,16),context_mask=torch.ones(1,3,dtype=torch.bool),num_inference_steps=1)
    h.remove()
    if ah:ah.remove()
    assert len(seen)==1 and not aux and torch.isfinite(result['action']).all()


@pytest.mark.parametrize('variant',['boundary','auxiliary'])
def test_configs_and_resume(variant):
    base=load_config('configs/libero/train_meanflow_diagonal_fix.yaml')
    cfg=load_config(f'configs/libero/train_imf_{variant}.yaml')
    base.output_dir=cfg.output_dir
    for key in base:
        if key not in {'meanflow','model','output_dir'}:assert cfg[key]==base[key],key
    for key in base.model:
        if key!='meanflow':assert cfg.model[key]==base.model[key],key
    for key in base.meanflow:
        if key!='objective':assert cfg.meanflow[key]==base.meanflow[key],key
    current=OmegaConf.to_container(cfg.meanflow,resolve=True)
    for key,value in [('variant','boundary' if variant=='auxiliary' else 'auxiliary'),('auxiliary_depth',7),('auxiliary_loss_weight',2.)]:
        if variant=='boundary' and key!='variant':continue
        saved=copy.deepcopy(current);saved['imf'][key]=value
        with pytest.raises(ValueError):validate_experiment_resume(current,saved)


def test_auxiliary_copied_function_and_original_video_layer_mapping():
    m=tiny_model(MeanFlowLibero,layers=40);m.configure_meanflow(config('auxiliary'),{});m.eval()
    batch=inputs(m);condition,_=prepare_condition(m,batch,torch.randn(2,16,2,2,2))
    z=batch['action'];t=torch.tensor([.4,.8]);seen=[]
    # A deliberately nonidentity mapping catches accidental 0..7 indexing in v.
    m.mot._video_source_layer=lambda i:(seen.append(i) or i//2)
    with torch.no_grad():
        u=m.predict_action_velocity(z,t,t,condition);u_mapping=seen.copy();seen.clear()
        v=m.predict_action_velocity_with_cache(latents_action=z,timestep_action=t*1000,timestep_end=t*1000,action_branch='aux_v',**condition)
    assert seen==u_mapping==list(range(40))
    torch.testing.assert_close(u,v,rtol=0,atol=0)


@pytest.mark.parametrize('variant',['boundary','auxiliary'])
@pytest.mark.parametrize('rank',['0','1'])
def test_production_launcher_from_other_directory(tmp_path,variant,rank):
    import os,subprocess,sys,json
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHON_BIN=sys.executable,PET_NNODES='2',PET_NPROC_PER_NODE='8',PET_NODE_RANK=rank,PET_MASTER_ADDR='10.1.1.1')
    for key in ['NNODES','NPROC_PER_NODE','NODE_RANK','BATCH_SIZE','WORLD_SIZE']:env.pop(key,None)
    p=subprocess.run(['bash',str(repo/f'scripts/libero/train_imf_{variant}.sh'),'--dry-run'],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    report=json.loads(p.stdout)
    assert report['world_size']==16 and report['micro_batch']==16 and report['global_batch']==256
    assert report['gradient_accumulation_steps']==1 and report['num_workers']==4
    assert report['action_layers']==40 and report['auxiliary_layers']==(8 if variant=='auxiliary' else 0)
    assert report['auxiliary_parameters']>0 if variant=='auxiliary' else report['auxiliary_parameters']==0
    assert report['interval']['diagonal_time_sampling']=='independent'
