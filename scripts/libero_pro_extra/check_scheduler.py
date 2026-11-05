import importlib.util,json,tempfile,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
spec=importlib.util.spec_from_file_location('extra_dispatch',ROOT/'scripts/libero_pro_extra/distributed_scheduler.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
with tempfile.TemporaryDirectory() as tmp:
 d=Path(tmp);q.OUT=d;q.STORE=d/'queue.json';q.OLD=d/'old.json'
 s=json.loads((OUT/'queue.json').read_text())
 for j in s['jobs'].values():j['state']='WAIT_ASSETS'
 q.STORE.write_text(json.dumps(s));q.OLD.write_text(json.dumps(dict(models={})))
 (d/'asset_checks').symlink_to(OUT/'asset_checks',target_is_directory=True)
 slots=[('h200-a',0),('h200-a',1),('h200-b',0),('h200-b',1)]
 with ThreadPoolExecutor(max_workers=4) as pool:r=list(pool.map(lambda x:q.claim(*x),slots))
 assert all(r) and len({x[0] for x in r})==4
 assert all(x[2] is False for x in r)
 for pair,item in zip(slots,r):assert q.claim(*pair)[0]==item[0] and q.claim(*pair)[2] is True
 q.OLD.write_text(json.dumps(dict(models={'old':dict(state='READY')})))
 assert q.claim('h200-a',0) is None
 q.OLD.write_text(json.dumps(dict(models={'old':dict(state='RUNNING',node='h200-a',gpu=0)})))
 assert q.claim('h200-a',0) is None and q.claim('h200-b',1) is not None
(out:=OUT/'scheduler-validation.json').write_text(json.dumps(dict(status='PASS',checks=['four distinct concurrent claims','same-slot resume','old READY jobs prioritized','old RUNNING GPU excluded','other GPUs unblocked']),indent=2)+'\n')
print('SCHEDULER_PASS')
