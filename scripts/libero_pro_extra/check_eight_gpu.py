"""Exercise mixed old/new controllers sharing one eight-slot queue."""
import importlib.util,json,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
modules=[]
for i,filename in enumerate(('distributed_scheduler.py','distributed_scheduler_8gpu.py')):
 spec=importlib.util.spec_from_file_location(f'extra_dispatch_{i}',ROOT/'scripts/libero_pro_extra'/filename);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);modules.append(q)
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp)
 for q in modules:q.OUT=d;q.STORE=d/'queue.json';q.OLD=d/'old.json'
 s=json.loads((OUT/'queue.json').read_text())
 for j in s['jobs'].values():
  j['state']='READY'
  for key in ('node','gpu','pid'):j.pop(key,None)
 (d/'queue.json').write_text(json.dumps(s));(d/'old.json').write_text(json.dumps(dict(models={})))
 (d/'asset_checks').symlink_to(OUT/'asset_checks',target_is_directory=True)
 slots=[(modules[0] if n in ('h200-a','h200-b') else modules[1],n,g) for n in ('h200-a','h200-b','h200-c','h200-d') for g in (0,1)]
 with ThreadPoolExecutor(max_workers=8) as pool:claims=list(pool.map(lambda x:x[0].claim(x[1],x[2]),slots))
 assert all(claims) and len({x[0] for x in claims})==8
 for (q,n,g),item in zip(slots,claims):assert q.claim(n,g)[0]==item[0] and q.claim(n,g)[2]
 (d/'old.json').write_text(json.dumps(dict(models={'old':dict(state='READY')})))
 assert modules[0].claim('h200-a',0) is None
 assert modules[1].claim('h200-c',0) is not None
 (d/'old.json').write_text(json.dumps(dict(models={'old':dict(state='RUNNING',node='h200-a',gpu=0)})))
 assert modules[0].claim('h200-a',0) is None
 assert modules[1].claim('h200-d',1) is not None
(OUT/'eight-gpu-scheduler-validation.json').write_text(json.dumps(dict(status='PASS',checks=['8 distinct concurrent claims across mixed controller versions','same-slot resume','A/B old evaluation exclusion','C/D immediately available independently']),indent=2)+'\n')
print('EIGHT_GPU_SCHEDULER_PASS')
