import csv,datetime,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103'
s=json.loads((OUT/'queue.json').read_text());rows=[]
for name,m in s['models'].items():
 row=dict(model=name,tokens=m['tokens'],checkpoint_step=30000,episodes=0,successes=0,complete=True)
 for axis in ('public800','object','lan','env'):
  j=s['jobs'][name+'__'+axis];n=j.get('episodes',0);k=j.get('successes',0)
  row[axis]=f"{k}/{n} ({j['state']}; expected {j['expected_episodes']})"
  row['episodes']+=n;row['successes']+=k;row['complete'] &= j['state']=='COMPLETE'
 row['success_rate']=row['successes']/2000 if row['complete'] else None
 rows.append(row)
report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=rows)
p=OUT/'model-evaluation-table.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
with (OUT/'model-evaluation-table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows:print(row['model'],row['episodes'],'/2000',row['success_rate'])
