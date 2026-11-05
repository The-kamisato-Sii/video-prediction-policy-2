"""Materialize eight inspectable high-priority 16-H200 commands and config diffs."""
import json,shlex,subprocess
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero_flow_meanflow.runtime import load_config,create_dataset
from vpp2.libero_flow.sampler import HomogeneousTaskSampler
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/libero_flow_meanflow_submit_20261104';OUT.mkdir(exist_ok=True)
base=json.loads((ROOT/'runs/libero_flow_all_submit_20261101/payload_32.json').read_text())
original=OmegaConf.load(ROOT/'runs/libero_flow/libero-flow32-a1of8-job-8b5523a1-90c5-44a0-842d-4bb7fdbe90fe-round0/resolved_config.yaml')
recon=OmegaConf.load(ROOT/'runs/libero_flow/libero-recon32-mse100-a1of8-job-026d46fa-8e3b-47c1-b78f-39939a3a3d8d-round0/resolved_config.yaml')
def flatten(d,p=''):
 out={}
 for k,v in d.items():
  key=f'{p}.{k}' if p else k
  if isinstance(v,dict):out.update(flatten(v,key))
  else:out[key]=v
 return out
reports=[]
for mode,objective in [('both','latent_meanflow'),('recon','direct_reconstruction')]:
 for variant in ('adaptive','diagonal','imf_boundary','imf_auxiliary'):
  key=f'{mode}-{variant}';name=f'libero-mf32-{key}-a1of8'
  settings=[f'flow_meanflow_variant={variant}',f'flow.objective={objective}',
    'flow.tokens=32','flow.action_denominator=8','flow.dt=4','batch_size=16','gradient_accumulation_steps=1',
    'flow.reconstruction_loss=mse','flow.scene_flow_loss_weight=100',
    f'flow.available_manifest={ROOT}/runs/libero_flow_submit_20261025/flow_manifest.json']
  cfg=load_config(settings)
  ref=original if mode=='both' else recon
  before=flatten(OmegaConf.to_container(ref,resolve=True));after=flatten(OmegaConf.to_container(cfg,resolve=True))
  diff={k:[before.get(k),after.get(k)] for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
  # Only objective/inference/output metadata may differ; optimizer, data,
  # backbone, preprocessing, cadence and all batch/sampler knobs must match.
  allowed=('meanflow.','inference.','wandb.','flow_meanflow_variant','output_dir','preflight.output_file','flow.objective','flow.motion_source')
  if mode=='both':allowed+=('flow.reconstruction_loss','flow.scene_flow_loss_weight')
  bad={k:v for k,v in diff.items() if not k.startswith(allowed)}
  assert not bad,bad
  assert cfg.max_steps==30000 and cfg.reference_global_batch_size==256
  OmegaConf.save(cfg,OUT/f'config_{key}.yaml')
  body=f'''set -euo pipefail
cd {ROOT}
source scripts/libero_flow/activate_env.sh
source scripts/libero/topology.sh
[[ "$NNODES" == 2 && "$NPROC_PER_NODE" == 8 && "$WORLD_SIZE" == 16 ]] || {{ echo "Expected 2 nodes x 8 GPUs"; exit 2; }}
export WANDB_MODE=offline
export RUN_NAME="{name}-${{TRAIN_JOB_ID:?}}-round${{RUNNING_ROUND:-0}}"
log_dir="runs/libero_flow_meanflow/${{RUN_NAME}}_logs"
mkdir -p "$log_dir"
exec 9>"$log_dir/node${{NODE_RANK}}.lock"
flock -n 9 || {{ echo "Node launcher already running"; exit 2; }}
# Fresh diagnostic process; formal training restarts from seed42 and action_init.
RUN_NAME="${{RUN_NAME}}-smoke" bash scripts/libero_flow_meanflow/train_{mode}_{variant}.sh \\
  --micro-batch 16 --diagnostic max_steps=8 eval_every=0 save_every=0 \\
  save_final_checkpoint=false wandb.enabled=false log_every=1 \\
  > >(tee -a "$log_dir/smoke-node${{NODE_RANK}}.log") 2>&1
bash scripts/libero_flow_meanflow/train_{mode}_{variant}.sh \\
  --micro-batch 16 max_steps=30000 warmup_steps=500 save_every=5000 eval_every=200 \\
  wandb.enabled=true wandb.mode=offline wandb.project=VPP2-LIBERO-FLOW \\
  "wandb.name=$RUN_NAME" "wandb.group=$RUN_NAME" \\
  > >(tee -a "$log_dir/node${{NODE_RANK}}.log") 2>&1
'''
  payload=dict(base,name=f'vpp2-mf32-{key}-a1of8-16h200-1104'.replace('_','-'),command='bash -c '+shlex.quote(body))
  (OUT/f'payload_{key}.json').write_text(json.dumps(payload,indent=2)+'\n')
  (OUT/f'command_{key}.sh').write_text(payload['command']+'\n')
  subprocess.run(['bash','-n',str(OUT/f'command_{key}.sh')],check=True)
  reports.append(dict(key=key,config_diff=diff,global_batch=256,micro_batch=16,accumulation=1,steps=30000,priority=10,gpus=16))
# Same class + exact settings = same record population and sampler sequence.
ds=create_dataset(cfg)
from vpp2.libero_flow.runtime import create_dataset as old_dataset
prior=old_dataset(original)
assert ds.records==prior.records and ds.ends==prior.ends
sampler=HomogeneousTaskSampler(ds,42,16,16,1)
ref_sampler=HomogeneousTaskSampler(prior,42,16,16,1)
assert list(sampler)==list(ref_sampler)
assert ds.audit['action_episodes']==214 and ds.audit['flow_episodes']==1589,ds.audit
(OUT/'config_validation.json').write_text(json.dumps(dict(status='PASS',experiments=reports,dataset=ds.audit,
  identical_records_and_sampler=True),indent=2)+'\n')
print('EIGHT_CONFIGS_AND_DATA_CONTROL_PASS',flush=True)
