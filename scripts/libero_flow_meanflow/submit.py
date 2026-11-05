"""Idempotent submission; require local control/math/full-backbone validations."""
import json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_flow_meanflow_submit_20261104'
cli=ROOT.parent/'tools/inspire-cli/bin/qz'
def call(args):
 p=subprocess.run([str(cli),*args],capture_output=True,text=True,check=True)
 d=json.loads(p.stdout)
 if d.get('ResponseMetadata',{}).get('Error'):raise RuntimeError(d['ResponseMetadata']['Error'])
 return d
assert json.loads((OUT/'config_validation.json').read_text())['status']=='PASS'
validation=ROOT/'runs/libero_flow_meanflow_validation_20261104'
for f in ['math.json','adapter.json','full_model_both.json','full_model_recon.json']:
 assert json.loads((validation/f).read_text())['status']=='PASS',f
reports=[]
for mode in ('both','recon'):
 for variant in ('adaptive','diagonal','imf_boundary','imf_auxiliary'):
  key=f'{mode}-{variant}';path=OUT/f'payload_{key}.json';payload=json.loads(path.read_text())
  result=call(['train','ListJobs','--workspace-id',payload['workspace_id'],'--keyword',payload['name'],'--page-number','1','--page-size','100'])['Result']
  matches=[j for j in result.get('jobs',[]) if j['name']==payload['name']]
  assert len(matches)<=1
  if matches:job_id=matches[0]['job_id']
  else:
   response=call(['train','CreateJob','--data','@'+str(path)])
   (OUT/f'response_{key}.json').write_text(json.dumps(response,indent=2)+'\n')
   job_id=response['Result']['job_id']
  response=call(['train','GetJob','--job-id',job_id]);d=response['Result']
  (OUT/f'detail_{key}.json').write_text(json.dumps(response,indent=2)+'\n')
  assert d['command']==payload['command']
  assert sum(c['instance_count']*c['gpu_count'] for c in d['framework_config'])==16
  assert int(d['priority_name'])==10
  row=dict(key=key,name=d['name'],job_id=job_id,status=d['status'],priority=10,gpus=16,
   tokens=32,action_fraction='1/8',flow_source='simulator_gt',flow_episodes=1589,
   global_batch=256,micro_batch=16,gradient_accumulation=1,max_steps=30000,
   action_nfe=1,flow_nfe=1,preflight_optimizer_steps=8)
  reports.append(row);(OUT/'jobs.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(row),flush=True)
