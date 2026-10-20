"""Single-GPU complete MeanFlow steps, including fresh Video prefill and AdamW updates.

Frozen Video remains BF16. Both modes use FP32 master parameters and AdamW state.
This isolates precision on one GPU; it does not simulate distributed ZeRO communication.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from vpp2.libero.runtime import create_action_model
from vpp2.meanflow.model import MeanFlowLibero, derivative_mode, prepare_condition, slice_condition
from vpp2.meanflow.objectives import data_loss, directional_derivative, sample_intervals


def cast_tree(x, dtype):
    # Preserve the context/proprio graph; do not detach the primary condition.
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: cast_tree(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [cast_tree(v, dtype) for v in x]
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', choices=['bf16', 'fp32_action'], required=True)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', mode=args.mode, batch_size=args.batch_size,
                  checkpoint=args.checkpoint, steps=[], video_dtype='bf16',
                  optimizer='AdamW FP32 masters and moments; fresh state, no ZeRO',
                  tf32=False, primary_backend='PyTorch default SDPA dispatch',
                  jvp_backend='MATH', workers=4, learning_rate=1e-4,
                  scope='Single GPU, includes dataloader wait/build_inputs/Video prefill/Action/JVP/backward/clip/optimizer')
    def save():
        tmp=out/'report.json.tmp'; tmp.write_text(json.dumps(report, indent=2)); tmp.replace(out/'report.json')
    def stage(s):
        report['stage']=s;save();print('STAGE',s,flush=True)
    start=time.perf_counter()
    try:
        torch.cuda.set_device(0)
        torch.manual_seed(42);random.seed(42);np.random.seed(42)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        cfg=OmegaConf.load(args.config)
        stage('model_construction')
        mc=OmegaConf.to_container(cfg.model, resolve=True)
        for k in ['_target_', 'meanflow', 'benchmark']:mc.pop(k, None)
        mc.update(load_text_encoder=False,action_dit_pretrained_path=None,skip_dit_load_from_pretrain=True)
        model=create_action_model(**mc,_model_class=MeanFlowLibero,model_dtype=torch.bfloat16,device='cuda:0')
        model.configure_meanflow(OmegaConf.to_container(cfg.meanflow,resolve=True),{'benchmark':True})
        stage('load_checkpoint')
        report['checkpoint_step']=model.load_checkpoint(args.checkpoint)
        model.eval().requires_grad_(False)
        model.mot.train();model.action_expert.train().requires_grad_(True)
        model.video_expert.eval().requires_grad_(False)
        model.proprio_encoder.train().requires_grad_(True)
        dtype=torch.bfloat16 if args.mode=='bf16' else torch.float32
        model.action_expert.to(dtype=dtype);model.proprio_encoder.to(dtype=dtype)
        model.proprio_encoder.register_forward_pre_hook(lambda m, ins: (ins[0].to(dtype),))
        params=list(model.action_expert.parameters())+list(model.proprio_encoder.parameters())
        assert len(set(map(id,params)))==len(params)
        assert all(p.requires_grad and p.dtype==dtype for p in params)
        # Match mixed-precision optimizer semantics, rather than BF16 Adam moments.
        masters=([torch.nn.Parameter(p.detach().float().clone()) for p in params]
                 if dtype==torch.bfloat16 else params)
        optimizer=torch.optim.AdamW(masters,lr=cfg.learning_rate,betas=tuple(cfg.optimizer_betas),
                                    weight_decay=cfg.weight_decay,foreach=False)
        stage('dataset')
        dc=OmegaConf.to_container(cfg.data.train,resolve=True);dc['decode_only_first_video_frame']=True
        dataset=instantiate(OmegaConf.create(dc))
        indices=[int(i*len(dataset)/(args.steps*args.batch_size)) for i in range(args.steps*args.batch_size)]
        report['dataset_indices']=indices
        loader=DataLoader(Subset(dataset,indices),batch_size=args.batch_size,num_workers=4,
                          pin_memory=True,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader)
        report['setup_seconds']=time.perf_counter()-start
        mf=model.meanflow_config
        report['loss_weighting']=mf.get('loss_weighting','uniform')
        report['adaptive_weighting']=mf.get('adaptive_weighting',{'power':.5,'c':.001})
        stage('training')
        for step in range(args.steps):
            # Identical stochastic inputs in corresponding steps, independent of dtype dispatch.
            torch.manual_seed(4200+step);torch.cuda.manual_seed_all(4200+step)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            t0=time.perf_counter();last=t0;timings={}
            def mark(name):
                nonlocal last
                torch.cuda.synchronize();now=time.perf_counter();timings[name]=now-last;last=now
            batch=next(iterator);mark('data_wait_seconds')
            inputs=model.build_inputs(batch,training_stage='action');mark('build_inputs_seconds')
            clean=inputs['action'];pad=inputs.get('action_loss_is_pad',inputs['action_is_pad'])
            latent=inputs['input_latents']
            if latent is not None:noise_video=torch.randn_like(latent)
            else:
                c=inputs['condition_latents'];frames=(inputs['num_video_frames']-1)//int(model.vae.temporal_downsample_factor)+1
                noise_video=torch.randn(c.shape[0],c.shape[1],frames,*c.shape[-2:],device=c.device,dtype=c.dtype)
            noise=torch.randn_like(clean)  # Common BF16 noise before FP32 promotion.
            condition,_=prepare_condition(model,inputs,noise_video)
            condition=cast_tree(condition,dtype)
            clean=clean.to(dtype);noise=noise.to(dtype)
            t,r=sample_intervals(clean.shape[0],clean.device,model.train_action_scheduler.shift,mf['interval']['diagonal_probability'])
            mark('video_prefill_and_cast_seconds')
            def predict(z,lo,hi):return model.predict_action_velocity(z,lo,hi,condition)
            def derivative(z,lo,hi,v,rows):
                fixed=slice_condition(condition,rows)
                with derivative_mode(model):
                    return directional_derivative(lambda x,a,b:model.predict_action_velocity(x,a,b,fixed),
                                                  z[rows],lo[rows],hi[rows],v[rows],mf['derivative_backend'])
            with torch.autocast('cuda',enabled=False):
                loss,metrics=data_loss(predict,derivative,clean,noise,r,t,pad,objective=mf['objective'],
                    baseline_reduction=mf.get('student_init',{}).get('format')=='baseline_init',
                    loss_weighting=mf.get('loss_weighting','uniform'),
                    adaptive_power=mf.get('adaptive_weighting',{}).get('power',.5),
                    adaptive_c=mf.get('adaptive_weighting',{}).get('c',.001))
                loss=loss*model.loss_lambda_action
            mark('action_forward_and_jvp_seconds')
            loss.backward();mark('backward_seconds')
            assert all(p.grad is None for p in model.video_expert.parameters())
            for param,master in zip(params,masters):
                master.grad=None if param.grad is None else param.grad.detach().float()
            grad_norm=torch.nn.utils.clip_grad_norm_(masters,float(cfg.max_grad_norm),error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for param,master in zip(params,masters):param.copy_(master)
            optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
            mark('clip_optimizer_copy_seconds')
            row=dict(step=step+1,loss=float(loss.detach()),grad_norm=float(grad_norm),
                     active_jvp_rows=int((t>r).sum()),seconds=time.perf_counter()-t0,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**timings)
            if not np.isfinite(row['loss']):raise FloatingPointError('Nonfinite loss')
            report['steps'].append(row);save();print('STEP',json.dumps(row),flush=True)
            del loss,metrics,condition,inputs,noise_video,clean,noise,batch
        steady=report['steps'][1:]
        report['summary']=dict(first_step_seconds=report['steps'][0]['seconds'],
            mean_last9_seconds=float(np.mean([x['seconds'] for x in steady])),
            median_last9_seconds=float(np.median([x['seconds'] for x in steady])),
            mean_all10_seconds=float(np.mean([x['seconds'] for x in report['steps']])),
            peak_allocated_gib=max(x['peak_allocated_gib'] for x in report['steps']),
            peak_reserved_gib=max(x['peak_reserved_gib'] for x in report['steps']))
        report['status']='PASS';stage('complete')
    except BaseException as e:
        report.update(status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'FAIL',error=repr(e))
        save();raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start;save()


if __name__=='__main__':main()
