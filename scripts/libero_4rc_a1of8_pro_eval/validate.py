import json
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero_flow.evaluate import validate_checkpoint
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_4rc32_a1of8_pro_eval_20261104'
s=json.loads((OUT/'queue.json').read_text());cfg=OmegaConf.load(OUT/'combined/public800/eval.yaml');train=OmegaConf.load(cfg.training_config)
assert train.flow.tokens==32 and train.flow.action_denominator==8
assert train.flow.objective=='latent_flow_matching' and train.flow.motion_source=='4rc_v1'
assert list(train.flow.motion_cameras)==['image']
assert train.flow.batch_sampling=='homogeneous_interleaved'
assert int(train.max_steps)==30000 and int(train.reference_global_batch_size)==256
r=validate_checkpoint(cfg)
assert r['flow_objective']=='latent_flow_matching' and r['all_weights_finite']
for axis in ('object','lan','env'):
 c=OmegaConf.load(OUT/f'combined/{axis}/eval.yaml');assert c.action_checkpoint==cfg.action_checkpoint and c.training_config==cfg.training_config
 gate=json.loads(Path(c.extra_asset_check).read_text());assert gate['status']=='PASS' and gate['checked_initial_states']==400
 assert all(c[k]==cfg[k] for k in ('dataset_stats','video_checkpoint','wan_model_dir','action_sampler','action_nfe','action_step','source_action_step'))
plan=json.loads((OUT/'partition-validation.json').read_text());assert plan['status']=='PASS' and plan['total_rollouts']==2000
prior=ROOT/'runs/libero_recon128_pro_eval_20261104'
audit=json.loads((prior/'shard-audit-validation.json').read_text());assert audit['status']=='PASS' and audit['total_fixture_episodes']==2000
old=json.loads((prior/'queue.json').read_text())
assert len(s['jobs'])==len(old['jobs'])==10
for newjob,oldjob in zip(s['jobs'].values(),old['jobs'].values()):
 a=json.loads(Path(newjob['config']).read_text());b=json.loads(Path(oldjob['config']).read_text())
 for k in ('shard_cells','shard_index','shard_count','action_nfe','action_sampler'):
  assert a[k]==b[k],k
(OUT/'validation.json').write_text(json.dumps(dict(status='PASS',checkpoint=r,partition=plan,shard_protocol_audit='PASS',flow_source='4rc_v1',strict_checkpoint_validation=True),indent=2)+'\n')
print('4RC_CHECKPOINT_AND_PARTITIONS_PASS',flush=True)
