import json,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103'
s=json.loads((OUT/'queue.json').read_text())
assert s['notebook_id'].split('-')[-1] in socket.gethostname()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used','--format=csv,noheader,nounits'],text=True)
gpus=[l.split(',') for l in rows.splitlines()];assert len(gpus)==8
assert all(int(r[2])<2048 for r in gpus),'GPU already busy; inspect existing workload before launching'
import torch,mujoco
from importlib.metadata import version
assert torch.cuda.device_count()==8 and mujoco.__version__=='3.3.2'
versions={k:version(k) for k in ('torch','diffusers','rpent-liberopro','rpent-libero','robosuite')}
assert versions['torch']=='2.11.0+cu128' and versions['diffusers']=='0.31.0'
assert versions['rpent-liberopro']=='0.2.0' and versions['robosuite']=='1.5.2'
checks=[]
for gpu in range(8):
 x=torch.ones((32,32),device=f'cuda:{gpu}',dtype=torch.bfloat16);y=x@x;torch.cuda.synchronize(gpu)
 assert y[0,0].item()==32
 checks.append(dict(gpu=gpu,name=torch.cuda.get_device_name(gpu),bf16_matmul='PASS'));del x,y
for j in s['jobs'].values():
 c=json.loads(Path(j['config']).read_text())
 for k in ('action_checkpoint','training_config','video_checkpoint','dataset_stats','wan_model_dir'):assert Path(c[k]).exists()
 for kind in ('bddl_files','init_files','assets'):assert (Path(c['benchmark_root'])/'liberopro/liberopro'/kind).exists()
report=dict(status='PASS',hostname=socket.gethostname(),time=time.time(),versions=versions,gpus=checks,all_model_paths_exist=True)
(OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
