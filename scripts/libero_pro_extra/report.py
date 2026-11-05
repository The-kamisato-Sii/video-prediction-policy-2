"""Write a denominator-explicit model table; incomplete axes never become final scores."""
import csv,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'

def main():
 s=json.loads((OUT/'queue.json').read_text());rows=[]
 for name,m in s['models'].items():
  row=dict(model=name,method=m['group'],action_fraction=m['action_fraction'],tokens=m['tokens'],action_nfe=m['config']['action_nfe'],checkpoint='30k' if Path(m['config']['action_checkpoint']).exists() else 'WAIT_30K')
  old=Path(m['config']['output_dir'])/'summary.json'
  if old.exists():
   r=json.loads(old.read_text());row['previous_public800']=f"{r['successes']}/{r['episodes']} ({100*r['success_rate']:.2f}%)" if r.get('complete') else f"PARTIAL {r.get('episodes',0)}/800"
  else:row['previous_public800']='pending'
  total=0;success=0;complete=True
  for axis in ('object','lan','env'):
   j=s['jobs'][name+'__'+axis];folder=OUT/'jobs'/(name+'__'+axis)
   report=folder/'evaluation/summary.json'
   if j['state']=='COMPLETE':
    r=json.loads(report.read_text());row[axis]=f"{r['successes']}/400 ({100*r['success_rate']:.2f}%)";total+=400;success+=r['successes']
    if axis=='env':
     for key in ('effective_environment_shift','unchanged_environment_controls'):
      x=r[key];row[key]=f"{x['successes']}/{x['episodes']} ({100*x['success_rate']:.2f}%)"
   else:
    complete=False;state=j['state']
    if state=='WAIT_ASSETS' and (OUT/'asset_checks'/f'{axis}.json').exists():state='QUEUED'
    row[axis]=f"{state} {j.get('episodes',0)}/400" + (f" {j['node']}:GPU{j['gpu']}" if state=='RUNNING' else '')
    total+=j.get('episodes',0);success+=j.get('successes',0)
  row['extra1200']=f'{success}/1200 ({100*success/1200:.2f}%)' if complete else f'PENDING {total}/1200'
  row['checkpoint_path']=m['config']['action_checkpoint'];rows.append(row)
 fields=['model','method','action_fraction','tokens','action_nfe','checkpoint','previous_public800','object','lan','env','effective_environment_shift','unchanged_environment_controls','extra1200','checkpoint_path']
 with (OUT/'model-evaluation-table.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (OUT/'model-evaluation-table.json').write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=rows),indent=2)+'\n')
 print('WROTE',OUT/'model-evaluation-table.csv')
 for r in rows:print(r['model'],r['previous_public800'],r['object'],r['lan'],r['env'])
if __name__=='__main__':main()
