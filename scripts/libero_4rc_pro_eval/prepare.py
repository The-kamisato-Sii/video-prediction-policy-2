import json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_4rc32_a1of2_pro_eval_20261104';OUT.mkdir(exist_ok=True)
if (OUT/'queue.json').exists():print('Preserving existing queue');raise SystemExit(0)
old=ROOT/'runs/libero_all_pro_eval_20261103';previous=json.loads((old/'queue.json').read_text())
job={'job_id':'job-7e676296-b932-46a3-b795-a8e93b9334b3'}
train=ROOT/f"runs/libero_flow/libero-4rc-flow32-a1of2-{job['job_id']}-round0"
checkpoint=train/'checkpoints/weights/step_030000.pt';assert checkpoint.is_file() and zipfile.is_zipfile(checkpoint)
jobs={};full_configs={};plans={}
for axis,parts in (('public800',4),('env',2),('object',2),('lan',2)):
 source=ROOT/'runs/libero_all_pro_eval_20261103/jobs'/f'flow128-all__{axis}'/'eval.yaml'
 cfg=json.loads(source.read_text());cfg.update(action_checkpoint=str(checkpoint),training_config=str(train/'resolved_config.yaml'),action_step=30000,output_dir=str(OUT/'combined'/axis/'evaluation'),runtime_config_dir=str(OUT/'combined'/axis/'runtime'))
 full_configs[axis]=cfg
 folder=OUT/'combined'/axis;folder.mkdir(parents=True,exist_ok=True);(folder/'eval.yaml').write_text(json.dumps(cfg,indent=2)+'\n')
 axes=('swap','task') if axis=='public800' else (axis,)
 cells=[(None,base+'_'+a,t) for base in ('libero_spatial','libero_object','libero_goal','libero_10') for a in axes for t in range(10)]
 partitions=[cells[i::parts] for i in range(parts)]
 assert len(set(sum((list(x) for x in partitions),[])))==len(cells)
 for index,subset in enumerate(partitions):
  key=f'4rc-flow32-a1of2__{axis}_part{index}';folder=OUT/'jobs'/key;folder.mkdir(parents=True,exist_ok=True)
  c=dict(cfg,output_dir=str(folder/'evaluation'),runtime_config_dir=str(folder/'runtime'),shard_index=index,shard_count=parts,shard_cells=[list(x) for x in subset],axis_full_expected=800 if axis=='public800' else 400)
  (folder/'eval.yaml').write_text(json.dumps(c,indent=2)+'\n')
  jobs[key]=dict(model='4rc-flow32-a1of2',axis=axis,state='READY',config=str(folder/'eval.yaml'),expected_episodes=len(subset)*10,episodes=0,successes=0)
 assert all(len(x)*10==200 for x in partitions)
 plans[axis]=dict(tasks=len(cells),parts=parts,episodes=len(cells)*10)
s=dict(notebook_id=previous['notebook_id'],notebook_name=previous['notebook_name'],model='4rc-flow32-a1of2',checkpoint=str(checkpoint),training_config=str(train/'resolved_config.yaml'),jobs=jobs,gpus=8,total_rollouts=2000,partition_plan=plans)
(OUT/'queue.json').write_text(json.dumps(s,indent=2)+'\n');(OUT/'partition-validation.json').write_text(json.dumps(dict(status='PASS',no_duplicated_cells=True,all_expected_cells_covered=True,all_parts_200_episodes=True,plan=plans,total_rollouts=sum(j['expected_episodes'] for j in jobs.values())),indent=2)+'\n')
(OUT/'README.txt').write_text('4RC flow32 action1/2, final30k: unchanged Pro public800 + object400 + language400 + environment400.\n10 disjoint200-rollout shards share8 GPU slots. Every task retains all10 trials with the same original seeds and init indices; only task scheduling changes.\nOriginal checkpoint/training configuration, strict tensor restoration, actionNFE10, video10k, exact image/proprio processing and first-real-observation training sampler parity are retained.\nShard reports are explicitly partial protocols; combined reports are validated again against full original800/400-cell protocols.\nThe existing isolated per-GPU idle guard remains active after evaluations complete.\n')
print('Prepared 4RC flow32 10 shards, 2000 rollouts')
