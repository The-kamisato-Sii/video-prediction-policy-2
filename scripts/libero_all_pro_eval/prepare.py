"""Three All-action+flow final models: public800 + extra1200 each."""
import json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103';OUT.mkdir(exist_ok=True)
if (OUT/'queue.json').exists():
 print('Queue already prepared; preserving state');raise SystemExit(0)
source=ROOT/'runs/libero_flow_all_submit_20261101'
base=json.loads((ROOT/'runs/libero_flow_eval_20261030/flow32-a1of4/eval.yaml').read_text())
extra=ROOT/'runs/libero_pro_extra_20261030'
models={};jobs={}
for item in json.loads((source/'jobs.json').read_text()):
 tokens=item['tokens'];name=f'flow{tokens}-all';train=ROOT/f"runs/libero_flow/libero-flow{tokens}-all-{item['job_id']}-round0"
 checkpoint=train/'checkpoints/weights/step_030000.pt'
 assert checkpoint.is_file() and zipfile.is_zipfile(checkpoint),checkpoint
 c=dict(base,action_checkpoint=str(checkpoint),training_config=str(train/'resolved_config.yaml'),action_step=30000,action_nfe=10)
 models[name]=dict(tokens=tokens,action_fraction='all',flow_fraction='all valid simulator GT',checkpoint=str(checkpoint),training_config=str(train/'resolved_config.yaml'))
 for axis in ('public800','object','lan','env'):
  key=name+'__'+axis;folder=OUT/'jobs'/key;folder.mkdir(parents=True,exist_ok=True)
  cfg=dict(c,output_dir=str(folder/'evaluation'),runtime_config_dir=str(folder/'runtime'))
  if axis!='public800':
   gate=extra/'asset_checks'/f'{axis}.json';g=json.loads(gate.read_text());assert g['status']=='PASS' and g['checked_initial_states']==400
   cfg.update(protocol='extra1200_v1',benchmark_root=str(extra/'runtime'),extra_axis=axis,extra_model_kind='flow',extra_assets_definition=str(extra/'assets-definition.json'),extra_asset_check=str(gate))
  (folder/'eval.yaml').write_text(json.dumps(cfg,indent=2)+'\n')
  jobs[key]=dict(model=name,axis=axis,state='READY',config=str(folder/'eval.yaml'),expected_episodes=800 if axis=='public800' else 400,episodes=0,successes=0)
# Longest jobs start first, then round-robin model/axis tasks across remaining slots.
order=[f'flow{t}-all__public800' for t in (128,64,32)]
order += [f'flow{t}-all__{axis}' for axis in ('object','lan','env') for t in (128,64,32)]
s=dict(protocol='public800_plus_extra1200',models=models,jobs={k:jobs[k] for k in order},gpus=8,total_rollouts=6000)
(OUT/'queue.json').write_text(json.dumps(s,indent=2)+'\n')
(OUT/'README.txt').write_text('Evaluate final step30000 All-action+flow simulator-GT FM models, tokens32/64/128.\nEach model: unchanged public800 (swap+task) + object400 + language400 + environment400 =2000 rollouts.\nSame seed42, trials1..10, init indices1..10, settle15, actionchunk32, replan10, actionNFE10, frozenVideo10k and exact training image/proprio processing.\n8 independent GPU slots: longest public800 jobs first, then dynamic work stealing of400-rollout jobs.\nOriginal evaluation code and all former results remain untouched. First observation must pass exact training/inference parity.\nIdle GPU guard runs independently and remains after the queue finishes, as requested.\n')
print('Prepared',len(models),'models',len(jobs),'jobs',s['total_rollouts'],'rollouts')
