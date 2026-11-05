import fcntl,json,time
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero import evaluate as harness
from vpp2.libero_pro_extra import evaluate as extra
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_4rc32_a1of8_pro_eval_20261104'
def main():
 with (OUT/'report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);s=json.loads((OUT/'queue.json').read_text());axes={};total=successes=0;complete=True
  for axis in ('public800','object','lan','env'):
   parts={k:j for k,j in s['jobs'].items() if j['axis']==axis};done=all(j['state']=='COMPLETE' for j in parts.values())
   n=sum(j.get('episodes',0) for j in parts.values());good=sum(j.get('successes',0) for j in parts.values());expected=800 if axis=='public800' else 400
   if done:
    folder=OUT/'combined'/axis;root=folder/'evaluation';root.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(folder/'eval.yaml');source=[]
    for key,j in parts.items():
     part=OUT/'jobs'/key/'evaluation';r=json.loads((part/'summary.json').read_text())
     assert r['complete'] and r['episodes']==200 and not r['missing'] and not r['issues']
     assert (part/'full_run.status').read_text().strip()=='0'
     parity=json.loads((OUT/'jobs'/key/f"inference-parity-gpu{j['gpu']}.json").read_text())
     assert parity['status']=='PASS' and parity['max_abs_vs_training_sample_task']==0 and parity['all_saved_tensors_equal']
     for p in (part/'results').rglob('*_results.json'):
      dest=root/'results'/key/p.relative_to(part/'results');dest.parent.mkdir(parents=True,exist_ok=True)
      if not dest.exists():dest.symlink_to(p.resolve())
     status=root/'status'/f'{key}.status';status.parent.mkdir(exist_ok=True);status.write_text('0\n')
     source.append(dict(job=key,directory=str(part),gpu=j['gpu']))
    contract=dict(config=OmegaConf.to_container(cfg,resolve=True),jobs=list(parts),gpus=sorted({j['gpu'] for j in parts.values()}),source_shards=source)
    contract_path=root/'RUN_CONFIG.json'
    if contract_path.exists():assert json.loads(contract_path.read_text())==contract
    else:contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    r=harness.audit(cfg) if axis=='public800' else extra.audit(cfg)
    assert r['complete'] and r['episodes']==expected and not r['issues'] and not r['missing'],r
    (root/'summary.json').write_text(json.dumps(r,indent=2)+'\n');(root/'full_run.status').write_text('0\n')
    n=r['episodes'];good=r['successes']
   axes[axis]=dict(complete=done,episodes=n,expected_episodes=expected,successes=good,success_rate=good/n if done else None)
   total+=n;successes+=good;complete &= done
  r=dict(model=s['model'],checkpoint_step=30000,action_fraction='1/8',tokens=32,objective='latent_flow_matching',flow_source='4rc_v1',complete=complete,episodes=total,expected_episodes=2000,successes=successes,success_rate=successes/2000 if complete else None,axes=axes,updated_time=time.time())
  p=OUT/'summary.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)
  print(s['model'],total,'/2000',r['success_rate'],flush=True)
if __name__=='__main__':main()
