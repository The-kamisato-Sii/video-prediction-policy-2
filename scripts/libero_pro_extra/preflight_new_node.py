"""Validate a new worker against the existing evaluation runtime."""
import argparse,json,os,socket,subprocess,sys,datetime
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args()
s=json.loads((OUT/'queue.json').read_text());assert s['nodes'][a.node]['notebook_id'].split('-')[-1] in socket.gethostname()
import torch,mujoco
from diffusers import AutoencoderKL
assert torch.cuda.device_count()==2
assert mujoco.__version__=='3.3.2'
versions={k:version(k) for k in ('torch','diffusers','rpent-liberopro','rpent-libero','robosuite')}
assert versions['torch']=='2.11.0+cu128' and versions['diffusers']=='0.31.0'
assert versions['rpent-liberopro']=='0.2.0' and versions['rpent-libero']=='0.2.0' and versions['robosuite']=='1.5.2'
for key in ('video_checkpoint','training_config','action_checkpoint','dataset_stats','wan_model_dir'):
 assert Path(s['models']['baseline-a1of8']['config'][key]).exists()
for axis in ('object','lan','env'):
 r=json.loads((OUT/'asset_checks'/f'{axis}.json').read_text());assert r['status']=='PASS' and r['checked_initial_states']==400
from vpp2.libero_pro_extra.runtime import register_environment_suites
register_environment_suites()
from libero.libero import benchmark
registry=benchmark.get_benchmark_dict()
for base in ('libero_spatial','libero_object','libero_goal','libero_10'):
 for axis in ('object','lan','env'):assert registry[base+'_'+axis]().n_tasks==10
checks=[]
for gpu in range(2):
 x=torch.ones((32,32),device=f'cuda:{gpu}',dtype=torch.bfloat16);y=x@x;torch.cuda.synchronize(gpu);assert float(y[0,0])==32
 checks.append(dict(gpu=gpu,name=torch.cuda.get_device_name(gpu),memory_bytes=torch.cuda.get_device_properties(gpu).total_memory,bf16_matmul='PASS'));del x,y
report=dict(status='PASS',node=a.node,hostname=socket.gethostname(),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),versions=versions,mujoco=mujoco.__version__,gpus=checks,shared_assets='PASS',suite_registry='PASS')
(OUT/'nodes'/a.node/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
