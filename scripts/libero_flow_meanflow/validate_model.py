"""Real RGB/GT data, full backbone, JVP/backward, routing, one-step and reload."""
import argparse,gc,json,time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from vpp2.libero_flow_meanflow.runtime import load_config,create_model,create_dataset
from vpp2.libero_flow.data import collate_flow_action

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['both','recon'],required=True);p.add_argument('--output',required=True);p.add_argument('--batch',type=int,default=1);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
common=[f'flow.objective={"latent_meanflow" if args.mode=="both" else "direct_reconstruction"}',
 'flow.available_manifest=runs/libero_flow_submit_20261025/flow_manifest.json',
 'flow.reconstruction_loss=mse','flow.scene_flow_loss_weight=100']
cfg=load_config(common);set_seed(42)
ds=create_dataset(cfg)
batches={i:collate_flow_action([ds[ds.first_index(i)+j] for j in range(args.batch)]) for i in (0,1)}
(out/f'data_{args.mode}.json').write_text(json.dumps(ds.audit,indent=2)+'\n')
print('BUILD_FULL_MODEL',args.mode,flush=True)
model=create_model(cfg);print('MODEL_READY',flush=True)
rows=[]
for variant in ('adaptive','diagonal','imf_boundary','imf_auxiliary'):
 cfg=load_config(common+[f'flow_meanflow_variant={variant}'])
 model.configure_meanflow(cfg.meanflow)
 for indicator in (0,1):
  set_seed(47)  # includes off-diagonal JVP for B=1
  model.configure_trainability();model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
  print('FORWARD',variant,indicator,flush=True);start=time.time()
  loss,metrics=model(batches[indicator]);assert torch.isfinite(loss);loss.backward()
  grad=model.action_expert.blocks[0].ffn[0].weight.grad;assert grad is not None and grad.float().norm()>0
  assert model.flow_adapters.indicator.weight.grad[indicator].float().norm()>0
  assert not any(p.grad is not None for p in model.video_expert.parameters())
  if args.mode=='both':assert not any(p.grad is not None for p in model.flow_codec.parameters())
  if args.mode=='recon' and indicator==1:
   assert model.motion_queries.grad.float().norm()>0
   assert model.flow_codec.decoder.conv_out.weight.grad.float().norm()>0
  if variant=='imf_auxiliary' and (indicator==0 or args.mode=='both'):
   assert model.action_expert.aux_v_blocks[0].ffn[0].weight.grad.float().norm()>0
  missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing,missing
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  row=dict(variant=variant,indicator=indicator,loss=float(loss.detach()),metrics=metrics,
   backbone_grad=float(grad.float().norm()),peak_gib=torch.cuda.max_memory_allocated()/2**30,
   seconds=time.time()-start,shapes=dict(model.last_shapes[str(indicator)]))
  print('BACKWARD_PASS',json.dumps(row),flush=True)
  model.zero_grad(set_to_none=True);del loss;gc.collect();model.eval()
  set_seed(42)
  result=model.sample_task(batches[indicator],indicator=indicator,steps=1)
  assert all(torch.isfinite(v).all() for v in result.values())
  row['output_shapes']={k:list(v.shape) for k,v in result.items()}
  assert result['action'].shape==(args.batch,32,7) if indicator==0 else result['scene_flow'].shape==(args.batch,1,3,8,224,224)
  rows.append(row);del result
  (out/f'progress_{args.mode}.json').write_text(json.dumps(rows,indent=2)+'\n')
# Checkpoint must contain auxiliary heads + learned decoder/queries where relevant.
model.eval();set_seed(42);before=model.sample_task(batches[0],0,1)['action'].cpu()
path=out/f'test_{args.mode}.pt';model.save_flow_checkpoint(path,step=0,resolved_cfg=cfg)
with torch.no_grad():model.flow_adapters.action_head[-1].bias.add_(1)
model.load_checkpoint(path);set_seed(42);after=model.sample_task(batches[0],0,1)['action'].cpu()
assert torch.equal(before,after)
path.unlink()
report=dict(status='PASS',mode=args.mode,batch=args.batch,branches=rows,checkpoint_roundtrip_exact=True)
(out/f'full_model_{args.mode}.json').write_text(json.dumps(report,indent=2)+'\n')
print('FULL_MODEL_PASS',args.mode,flush=True)
