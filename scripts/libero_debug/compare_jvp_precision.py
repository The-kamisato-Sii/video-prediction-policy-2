"""Paired precision diagnostic on real LIBERO inputs; no optimizer updates."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from vpp2.libero.runtime import create_action_model
from vpp2.meanflow.model import MeanFlowLibero, derivative_mode, prepare_condition
from vpp2.meanflow.objectives import directional_derivative, expand_time, masked_per_sample


def tree_cast(x, dtype):
    if torch.is_tensor(x):
        return x.detach().to(dtype=dtype) if x.is_floating_point() else x.detach()
    if isinstance(x, dict):
        return {k: tree_cast(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [tree_cast(v, dtype) for v in x]
    return x


def comparison(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    denom = b.norm().clamp_min(1e-30)
    return dict(relative_l2=float((a-b).norm()/denom),
                cosine=float(torch.dot(a,b)/(a.norm()*denom).clamp_min(1e-30)),
                max_abs=float((a-b).abs().max()))


def norm(x):
    return dict(l2=float(x.float().norm()), rms=float(x.float().square().mean().sqrt()),
                max_abs=float(x.float().abs().max()), finite=bool(torch.isfinite(x).all()))


def gradient_report(model, reference=None):
    square = torch.zeros((), device='cuda', dtype=torch.float64)
    ref_square = square.clone(); delta = square.clone(); dot = square.clone()
    result = {} if reference is None else None
    for name, p in model.action_expert.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            raise FloatingPointError(f'Nonfinite gradient: {name}')
        square += g.square().sum().double()
        if reference is None:
            result[name] = p.grad.detach().cpu().clone()
        else:
            r = reference[name].to(device=g.device, dtype=torch.float32)
            ref_square += r.square().sum().double()
            delta += (g-r).square().sum().double()
            dot += (g*r).sum().double()
    report = dict(preclip_norm=float(square.sqrt()))
    if reference is not None:
        report.update(relative_l2_to_bf16=float((delta/ref_square.clamp_min(1e-30)).sqrt()),
                      cosine_to_bf16=float(dot/(square*ref_square).sqrt().clamp_min(1e-30)))
    return report, result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--batch-size',type=int,default=2)
    args=p.parse_args()
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',checkpoint=str(Path(args.checkpoint).resolve()),
                seed=42,batch_size=args.batch_size,results=[],finite_differences=[],
                scope='Fixed BF16-quantized weights and frozen BF16 Video cache; precision comparison, not convergence proof',
                gradient_scope='All Action parameters; conditioning and proprio output held fixed',
                primary_backend='MATH for all modes to isolate precision',tf32=False)
    def save():
        tmp=root/'report.json.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'report.json')
    def stage(s):
        report['stage']=s;save();print('STAGE',s,flush=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.cuda.set_device(0)
    started=time.perf_counter()
    try:
        stage('construct_model')
        cfg=OmegaConf.load(args.config)
        mc=OmegaConf.to_container(cfg.model,resolve=True)
        for key in ['_target_','meanflow','benchmark']:
            mc.pop(key,None)
        mc.update(load_text_encoder=False,action_dit_pretrained_path=None,skip_dit_load_from_pretrain=True)
        model=create_action_model(**mc,_model_class=MeanFlowLibero,model_dtype=torch.bfloat16,device='cuda:0')
        model.configure_meanflow(OmegaConf.to_container(cfg.meanflow,resolve=True),{'diagnostic':True})
        stage('load_checkpoint')
        step=model.load_checkpoint(args.checkpoint)
        report['checkpoint_step']=step
        model.eval().requires_grad_(False)
        model.action_expert.train().requires_grad_(True)
        model.mot.train()
        # Keep Video in its frozen evaluation mode after mot.train() traverses aliases.
        model.video_expert.eval().requires_grad_(False)
        assert model.mot.mixtures['action'] is model.action_expert
        stage('real_dataset_and_fixed_condition')
        dc=OmegaConf.to_container(cfg.data.train,resolve=True)
        dc['decode_only_first_video_frame']=True
        ds=instantiate(OmegaConf.create(dc))
        indices=[int(i*len(ds)/args.batch_size) for i in range(args.batch_size)]
        batch=default_collate([ds[i] for i in indices]);del ds
        torch.save(batch,root/'batch.pt')
        report['dataset_indices']=indices
        with torch.no_grad():
            inputs=model.build_inputs(batch,training_stage='action')
            clean=inputs['action'].detach()
            pad=inputs.get('action_loss_is_pad',inputs['action_is_pad'])
            latent=inputs['input_latents']
            if latent is not None:
                noise_video=torch.randn_like(latent)
            else:
                c=inputs['condition_latents'];frames=(inputs['num_video_frames']-1)//int(model.vae.temporal_downsample_factor)+1
                noise_video=torch.randn(c.shape[0],c.shape[1],frames,*c.shape[-2:],device=c.device,dtype=c.dtype)
            condition,_=prepare_condition(model,inputs,noise_video)
            condition=tree_cast(condition,torch.bfloat16)
            noise=torch.randn_like(clean)
        torch.save(dict(clean=clean.cpu(),noise=noise.cpu(),noise_video=noise_video.cpu(),pad=pad.cpu()),root/'fixed_noise.pt')
        report['condition_video_seq_len']=condition['video_seq_len']
        report['action_shape']=list(clean.shape)
        report['action_checkpoint_execution_dtype']='BF16 weights promoted for FP32 cases; no master-weight recovery'
        report['load_and_setup_seconds']=time.perf_counter()-started
        # Conditions include the same detached proprio token in every variant.
        condition32=tree_cast(condition,torch.float32)
        audit=dict(active=False,linear_calls=0)
        def audit_linear(module, ins):
            if audit['active']:
                assert module.weight.dtype==torch.float32
                assert ins[0].dtype==torch.float32, (type(module).__name__,ins[0].dtype)
                audit['linear_calls']+=1
        audit_handles=[m.register_forward_pre_hook(audit_linear) for m in model.action_expert.modules() if isinstance(m,torch.nn.Linear)]
        v=noise.float()-clean.float()
        grids=[('low',(.2,.3),(.05,.1)),('mid',(.55,.65),(.2,.3)),
               ('high',(.9,.95),(.3,.4)),('top',(.99,.999),(.4,.5))]
        if args.batch_size!=2:
            raise ValueError('This fixed time grid requires batch-size=2')
        samples=[]
        for label,ts,rs in grids:
            t=torch.tensor(ts,device='cuda');r=torch.tensor(rs,device='cuda')
            z=(clean.float()+expand_time(t,clean)*v).to(clean)
            samples.append((label,z,r,t))
        references={};outputs={}
        for mode in ['bf16','fp32_jvp_only','fp32_action']:
            stage(mode)
            for label,z,r,t in samples:
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin=time.perf_counter()
                use32=mode!='bf16'
                model.action_expert.to(dtype=torch.float32 if use32 else torch.bfloat16)
                cd=condition32 if use32 else condition
                zd=z.float() if use32 else z
                def predict(x,lo,hi):return model.predict_action_velocity(x,lo,hi,cd)
                audit['active']=use32
                with torch.autocast('cuda',enabled=False),derivative_mode(model):
                    d=directional_derivative(predict,zd,r,t,v,'jvp')
                audit['active']=False
                torch.cuda.synchronize();jvp_seconds=time.perf_counter()-begin
                # Primary stays BF16 in the derivative-only variant.
                main32=mode=='fp32_action'
                model.action_expert.to(dtype=torch.float32 if main32 else torch.bfloat16)
                cp=condition32 if main32 else condition
                zp=z.float() if main32 else z
                target=v-expand_time(t-r,v)*d.detach()
                with torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                    pred=model.predict_action_velocity(zp,r,t,cp)
                    loss=masked_per_sample(pred,target.detach(),pad)[0].mean()
                    loss.backward()
                grad,stored=gradient_report(model,references.get(label))
                if mode=='bf16':references[label]=stored
                row=dict(mode=mode,time_bucket=label,t=t.tolist(),r=r.tolist(),loss=float(loss.detach()),
                         jvp=norm(d),correction=norm(expand_time(t-r,v)*d),target=norm(target),
                         gradient=grad,jvp_seconds_including_cast=jvp_seconds,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                torch.cuda.synchronize();row['total_seconds_including_gradient_comparison']=time.perf_counter()-begin
                if mode!='bf16':
                    old=outputs[('bf16',label)]
                    row['jvp_vs_bf16']=comparison(d,old['d'])
                    row['prediction_vs_bf16']=comparison(pred.detach(),old['pred'])
                    row['target_vs_bf16']=comparison(target,old['target'])
                    if mode=='fp32_jvp_only':
                        assert torch.equal(pred.detach(),old['pred']), 'Primary prediction changed in JVP-only comparison'
                outputs[(mode,label)]=dict(d=d.detach().clone(),pred=pred.detach().clone(),target=target.detach().clone())
                report['results'].append(row);save();print('RESULT',json.dumps(row),flush=True)
                del pred,loss,target,d
        stage('fp32_central_finite_difference')
        model.zero_grad(set_to_none=True);model.action_expert.float()
        assert all(p.dtype==torch.float32 for p in model.action_expert.parameters())
        report['fp32_action_parameter_dtype_verified']=True
        report['fp32_linear_input_checks']=audit['linear_calls']
        audit['active']=True
        for label,z,r,t in samples:
            with torch.autocast('cuda',enabled=False),derivative_mode(model):
                for h in [3e-4,1e-4,3e-5]:
                    plus=model.predict_action_velocity(z.float()+h*v,r,t+h,condition32).float()
                    minus=model.predict_action_velocity(z.float()-h*v,r,t-h,condition32).float()
                    fd=(plus-minus)/(2*h)
                    row=dict(time_bucket=label,h=h,fd=norm(fd),jvp_vs_fd=comparison(outputs[('fp32_action',label)]['d'],fd),
                             bf16_jvp_vs_fd=comparison(outputs[('bf16',label)]['d'],fd))
                    report['finite_differences'].append(row);save();print('FD',json.dumps(row),flush=True)
        assert all(p.grad is None for p in model.video_expert.parameters())
        report['video_gradient_absent']=True
        report['status']='PASS';stage('complete')
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));save();raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;save()


if __name__=='__main__':main()
