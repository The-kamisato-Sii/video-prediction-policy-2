"""Small 40-layer model through the actual BF16/ZeRO-1 trainer and full-state resume."""
import argparse
import types
from pathlib import Path
import torch
from accelerate import PartialState
from omegaconf import OmegaConf
from vpp2.meanflow.cli import load_config
from vpp2.meanflow.model import MeanFlowLibero
from vpp2.libero.trainer import LiberoTrainer
from test_meanflow import tiny_model
from trainer_meanflow import Dataset,build

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--resume');parser.add_argument('--steps',type=int,default=2)
    args=parser.parse_args();state=PartialState();torch.manual_seed(42)
    cfg=load_config(f'configs/libero/train_imf_{args.variant}.yaml')
    cfg.output_dir=args.output;cfg.overwrite=False;cfg.batch_size=2;cfg.num_workers=0
    cfg.eval_every=0;cfg.save_every=2;cfg.log_every=1;cfg.wandb.enabled=False;cfg.resume=args.resume
    m=tiny_model(MeanFlowLibero,layers=40,device=str(state.device),dtype=torch.bfloat16,checkpoint=True)
    def build_bf16(self,sample,**kwargs):
        sample={k:v.to(dtype=self.torch_dtype) if v.is_floating_point() else v for k,v in sample.items()}
        return build(self,sample,**kwargs)
    m.build_inputs=types.MethodType(build_bf16,m);m.configure_meanflow(OmegaConf.to_container(cfg.meanflow,resolve=True),{'synthetic':True})
    video=Path(args.output)/'video.pt'
    with state.main_process_first():
        video.parent.mkdir(parents=True,exist_ok=True)
        if not video.exists():torch.save({'dit':m.video_expert.state_dict(),'step':10000},video)
    cfg.frozen_video_checkpoint=str(video)
    trainer=LiberoTrainer(m,Dataset(),cfg=cfg)
    assert trainer._deepspeed_zero_stage()==1
    grouped=[p for group in trainer.model.optimizer.bit16_groups for p in group]
    assert len(grouped)==len({id(p) for p in grouped})
    assert {id(p) for p in grouped}=={id(p) for p in m.parameters() if p.requires_grad}
    trainer.max_steps=args.steps;trainer.train();assert trainer.global_step==args.steps
    assert all(torch.isfinite(p).all() for p in m.action_expert.parameters())
    print('IMF_ZERO1_BF16_TRAINER_PASS',args.variant,args.steps,'resume',bool(args.resume),flush=True)
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
