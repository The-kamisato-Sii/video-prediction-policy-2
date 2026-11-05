"""Finish the missing final Flow64 1/2 public800 evaluation on two GPUs."""
import json,os,subprocess,sys,datetime,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_flow_eval_20261030/flow64-a1of2'
STATE=ROOT/'runs/libero_pro_extra_20261030/public800-backfill.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(s):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)
state=dict(model='flow64-a1of2',protocol='public800',gpus=[0,1],node='h200-a',hostname=socket.gethostname(),started_utc=now(),state='RUNNING',pid=os.getpid(),expected_episodes=800)
if STATE.exists():
 previous=json.loads(STATE.read_text())
 if previous.get('state')=='FAILED':state['previous_attempts']=[previous]
write(state)
env=dict(os.environ)
project=ROOT.parent
pro_env=Path(sys.executable).parent
env.update(PYTHONPATH=str(ROOT/'src'),PATH=str(pro_env)+':'+str(project/'tools/tmux-portable')+':'+env.get('PATH',''),LD_LIBRARY_PATH=str(project/'tools/libero-pro-glvnd/lib')+':'+env.get('LD_LIBRARY_PATH',''),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VPP2_TORCH_LOAD_MMAP='1')
for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
args=[sys.executable,'-u','-m','vpp2.libero_flow.evaluate','--config',str(OUT/'eval.yaml'),'--gpus','0','1']
if (OUT/'evaluation/RUN_CONFIG.json').exists():args.append('--resume')
try:
 with (OUT/'backfill-coordinator.log').open('a') as log:code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT).returncode
 assert code==0,code
 r=json.loads((OUT/'evaluation/summary.json').read_text());assert r['complete'] and r['episodes']==800 and not r['issues'] and not r['missing']
 for gpu in (0,1):
  p=json.loads((OUT/f'inference-parity-gpu{gpu}.json').read_text());assert p['status']=='PASS' and p['max_abs_vs_training_sample_task']==0
 state.update(state='COMPLETE',episodes=800,successes=r['successes'],success_rate=r['success_rate'],ended_utc=now())
 write(state)
 subprocess.run([sys.executable,str(ROOT/'scripts/libero_pro_extra/report.py')],check=True)
except Exception as e:
 state.update(state='FAILED',error=repr(e),ended_utc=now());write(state);raise
