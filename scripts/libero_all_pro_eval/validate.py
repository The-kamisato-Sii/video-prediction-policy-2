import json
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero_flow.evaluate import validate_checkpoint
from vpp2.libero_pro_extra.evaluate import validate_checkpoint as validate_extra
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103'
s=json.loads((OUT/'queue.json').read_text());reports=[]
for model,m in s['models'].items():
 cfg=OmegaConf.load(s['jobs'][model+'__public800']['config']);train=OmegaConf.load(cfg.training_config)
 assert train.flow.tokens==m['tokens'] and train.flow.action_denominator==1
 assert train.flow.objective=='latent_flow_matching'
 assert train.flow.batch_sampling=='homogeneous_interleaved' and train.flow.flow_episode_scope=='all'
 assert int(train.max_steps)==30000 and int(train.reference_global_batch_size)==256
 assert train.flow.get('motion_source','simulator_gt')=='simulator_gt'
 r=validate_checkpoint(cfg);reports.append(r)
 for axis in ('object','lan','env'):
  c=OmegaConf.load(s['jobs'][model+'__'+axis]['config'])
  assert c.action_checkpoint==cfg.action_checkpoint and c.training_config==cfg.training_config
  assert c.action_nfe==10 and c.action_step==30000
  gate=json.loads(Path(c.extra_asset_check).read_text());assert gate['status']=='PASS' and gate['checked_initial_states']==400
  assert all(c[k]==cfg[k] for k in ('dataset_stats','video_checkpoint','wan_model_dir','action_sampler','source_action_step'))
 assert sum(s['jobs'][model+'__'+a]['expected_episodes'] for a in ('public800','object','lan','env'))==2000
assert sum(j['expected_episodes'] for j in s['jobs'].values())==6000
(OUT/'validation.json').write_text(json.dumps(dict(status='PASS',models=reports,training_and_checkpoint_match=True,total_rollouts=6000,public_episodes=800,extra_episodes=1200),indent=2)+'\n')
print('ALL_THREE_FINAL_MODELS_VALIDATED',flush=True)
