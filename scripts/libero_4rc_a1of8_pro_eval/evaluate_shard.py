"""Partition task manifests only; reuse the original coordinators and worker paths."""
import argparse,json
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero import evaluate as harness
from vpp2.libero_flow import evaluate as flow
from vpp2.libero_pro_extra import evaluate as extra

def run(cfg,gpu,resume=False):
 subset=[(x[0],x[1],int(x[2])) for x in cfg.shard_cells]
 expected_full=harness.cells(cfg.benchmark) if cfg.protocol=='public800' else extra.cells(cfg)
 assert subset==expected_full[int(cfg.shard_index)::int(cfg.shard_count)]
 assert len(set(subset))==20 and len(subset)*10==200
 original_validate=harness.validate_config;original_extra_validate=extra.validate_config
 def validate(c):
  p=dict(original_extra_validate(c) if c.protocol=='extra1200_v1' else original_validate(c))
  p.update(episodes=len(subset)*p['trials'],name=p['name']+f'/part{cfg.shard_index}of{cfg.shard_count}')
  return p
 harness.cells=lambda benchmark:subset
 harness.validate_config=validate
 if cfg.protocol=='public800':
  harness.validate_checkpoint=flow.validate_checkpoint
  harness.worker_config=flow.worker_config
  harness.run(cfg,[gpu],resume=resume)
 else:
  extra.cells=lambda c:subset
  extra.validate_config=validate
  # Full environment/control totals are checked again in the combined audit.
  def audit(c):
   r=harness.audit(c);r.update(extra_axis=c.extra_axis,partial_task_partition=True)
   return r
  extra.audit=audit
  extra.run(cfg,gpu,resume=resume)
 report=Path(cfg.output_dir)/'summary.json';r=json.loads(report.read_text())
 r.update(partial_task_partition=True,shard_index=int(cfg.shard_index),shard_count=int(cfg.shard_count),axis_full_expected=int(cfg.axis_full_expected))
 report.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(OmegaConf.load(a.config),a.gpu,a.resume)
