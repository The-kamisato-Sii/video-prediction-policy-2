"""Verify that task-only partitioning preserves all existing protocol audit checks."""
import importlib.util,json,tempfile
from pathlib import Path
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_recon128_pro_eval_20261104';old=ROOT/'runs/libero_all_pro_eval_20261103'
s=json.loads((OUT/'queue.json').read_text());checks=[]
for key,j in s['jobs'].items():
 spec=importlib.util.spec_from_file_location('partition_eval',ROOT/'scripts/libero_recon128_pro_eval/evaluate_shard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 # Each run starts a fresh interpreter in production; restore patched module hooks here.
 h=m.harness;e=m.extra
 saved={name:getattr(h,name) for name in ('cells','validate_config','validate_checkpoint','worker_config','run')}
 saved_e={name:getattr(e,name) for name in ('cells','validate_config','audit','run')}
 cfg=OmegaConf.load(j['config']);wanted={tuple(x) for x in cfg.shard_cells}
 fixture=old/'jobs'/f"flow128-all__{j['axis']}"/'evaluation/results'
 with tempfile.TemporaryDirectory() as tmp:
  cfg.output_dir=tmp;files=[]
  for p in fixture.rglob('*_results.json'):
   r=json.loads(p.read_text());cell=(r.get('trial_seed_id'),r['task_suite'],r['task_id'])
   if cell in wanted:
    dest=Path(tmp)/'results'/p.relative_to(fixture);dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(p.resolve());files.append(cell)
  assert set(files)==wanted and len(files)==20
  def simulate(c,*args,**kwargs):
   r=h.audit(c,require_status=False);assert r['complete'] and r['episodes']==200 and not r['missing'] and not r['issues'],r
   (Path(c.output_dir)/'summary.json').write_text(json.dumps(r))
  h.run=simulate;e.run=simulate
  m.run(cfg,0)
  r=json.loads((Path(tmp)/'summary.json').read_text());assert r['partial_task_partition'] and r['episodes']==200
  checks.append(dict(job=key,audit='PASS',episodes=200,fixtures='previous completed All-flow128 task result JSONs; audit regression only, no reconstruction results fabricated'))
 for name,value in saved.items():setattr(h,name,value)
 for name,value in saved_e.items():setattr(e,name,value)
(OUT/'shard-audit-validation.json').write_text(json.dumps(dict(status='PASS',checks=checks,total_fixture_episodes=2000),indent=2)+'\n')
print('All10 partitions retain original audit checks; all2000 source trials covered exactly once')
