"""Check extra-protocol auditing rejects missing episodes, wrong seeds and duplicates."""
from pathlib import Path
import sys,json,tempfile,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from omegaconf import OmegaConf
from vpp2.libero_pro_extra.evaluate import audit,cells,validate_config
from vpp2.libero.eval_helpers import resolve_max_steps
out=ROOT/'runs/libero_pro_extra_20261030';passed=[]
for axis in ('object','lan','env'):
 cfg=OmegaConf.load(out/'jobs'/f'baseline-a1of8__{axis}'/'eval.yaml')
 with tempfile.TemporaryDirectory() as tmp:
  cfg.output_dir=tmp;p=Path(tmp);(p/'results').mkdir();(p/'status').mkdir()
  (p/'RUN_CONFIG.json').write_text(json.dumps(dict(config=OmegaConf.to_container(cfg,resolve=True),jobs=['test'])))
  (p/'status/test.status').write_text('0\n')
  for _,suite,tid in cells(cfg):
   r=dict(task_suite=suite,task_id=tid,trial_seed_id=None,success_episodes=list(range(5)),failure_episodes=list(range(5,10)),total_episodes=10,successes=5,
          reset_mode='task_init_state',trial_seed_mode='benchmark_seed_sequence',policy_seed=42,max_steps_per_episode=resolve_max_steps(suite),init_state_indices=list(range(1,11)),trial_seeds=list(range(1,11)))
   (p/'results'/f'{suite}_{tid}_results.json').write_text(json.dumps(r))
  r=audit(cfg);assert r['complete'] and r['episodes']==400 and r['successes']==200
  if axis=='env':assert r['effective_environment_shift']['episodes']==350 and r['unchanged_environment_controls']['episodes']==50
  sample=next((p/'results').glob('*_results.json'));original=sample.read_text();bad=json.loads(original);bad['trial_seeds'][0]=999;sample.write_text(json.dumps(bad));assert not audit(cfg)['complete'];sample.write_text(original)
  dup=p/'results/duplicate_results.json';dup.write_text(original);assert not audit(cfg)['complete'];dup.unlink()
  sample.unlink();assert not audit(cfg)['complete']
  passed.append(axis)
assert all(len(cells(OmegaConf.load(out/'jobs'/f'baseline-a1of8__{a}'/'eval.yaml')))==40 for a in passed)
(out/'protocol-validation.json').write_text(json.dumps(dict(status='PASS',axes=passed,checks=['400 episodes per axis','wrong seed rejected','duplicate rejected','missing cell rejected','environment 350+50 split']),indent=2)+'\n')
print('PROTOCOL_AUDIT_PASS',passed)
