"""Eight independent one-GPU evaluations with persistent, locked task claims."""
import datetime,fcntl,json,os,socket,subprocess,time
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_4rc32_a1of4_pro_eval_20261104';STORE=OUT/'queue.json';LAUNCH=ROOT/'scripts/libero_4rc_a1of4_pro_eval/run_job.sh'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextmanager
def transaction():
 with (OUT/'queue.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);s=json.loads(STORE.read_text());yield s
  s['updated_utc']=now();p=OUT/f'.queue.{os.getpid()}.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STORE)
def claim(gpu):
 with transaction() as s:
  # On restart, resume a slot's interrupted work on its original physical GPU.
  for key,j in s['jobs'].items():
   if j['state']=='RUNNING' and j.get('gpu')==gpu:return key
  for key,j in s['jobs'].items():
   if j['state']=='READY':
    j.update(state='RUNNING',gpu=gpu,started_utc=now(),hostname=socket.gethostname());return key
 return None
def progress(key):
 episodes=successes=0
 for p in (OUT/'jobs'/key/'evaluation/results').rglob('*_results.json'):
  try:
   r=json.loads(p.read_text());episodes+=len(r['success_episodes'])+len(r['failure_episodes']);successes+=len(r['success_episodes'])
  except (ValueError,KeyError,OSError):continue
 with transaction() as s:s['jobs'][key].update(episodes=episodes,successes=successes,heartbeat_utc=now())
def audit(key,gpu):
 folder=OUT/'jobs'/key;r=json.loads((folder/'evaluation/summary.json').read_text())
 with transaction() as s:expected=s['jobs'][key]['expected_episodes']
 assert r['complete'] and r['episodes']==expected and not r['issues'] and not r['missing'],r
 assert (folder/'evaluation/full_run.status').read_text().strip()=='0'
 parity=json.loads((folder/f'inference-parity-gpu{gpu}.json').read_text())
 assert parity['status']=='PASS' and parity['max_abs_vs_training_sample_task']==0 and parity['all_saved_tensors_equal']
 return r
def alive(pid,key):
 try:
  p=Path(f'/proc/{pid}');stat=(p/'stat').read_text()
  return stat[stat.rfind(')')+2:].split()[0]!='Z' and str(LAUNCH).encode() in (p/'cmdline').read_bytes() and key.encode() in (p/'cmdline').read_bytes()
 except (FileNotFoundError,ProcessLookupError):return False
def slot(gpu):
 while True:
  key=claim(gpu)
  if key is None:return
  folder=OUT/'jobs'/key
  try:
   with transaction() as s:pid=s['jobs'][key].get('pid')
   if pid and alive(pid,key):
    while alive(pid,key):progress(key);time.sleep(15)
    code=int((folder/'exit.status').read_text())
   else:
    with (folder/'coordinator.log').open('a') as log:
     p=subprocess.Popen(['bash',str(LAUNCH),key,str(gpu)],stdout=log,stderr=subprocess.STDOUT)
     with transaction() as s:s['jobs'][key]['pid']=p.pid
     print(json.dumps(dict(event='START',job=key,gpu=gpu,pid=p.pid)),flush=True)
     while p.poll() is None:progress(key);time.sleep(15)
     code=p.returncode
   progress(key)
   if code:raise RuntimeError(f'Evaluation exit {code}; see coordinator/worker logs')
   r=audit(key,gpu)
   with transaction() as s:s['jobs'][key].update(state='COMPLETE',ended_utc=now(),successes=r['successes'],episodes=r['episodes'],success_rate=r['success_rate'])
   print(json.dumps(dict(event='COMPLETE',job=key,gpu=gpu)),flush=True)
  except Exception as e:
   with transaction() as s:s['jobs'][key].update(state='FAILED',error=repr(e),ended_utc=now())
   print(json.dumps(dict(event='FAILED',job=key,gpu=gpu,error=repr(e))),flush=True)
  subprocess.run(['python',str(ROOT/'scripts/libero_4rc_a1of4_pro_eval/report.py')],check=False)
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--single-gpu',type=int,choices=range(8));args=parser.parse_args()
 gpus=range(4,8) if args.single_gpu is None else [args.single_gpu]
 lock_name='controller.lock' if args.single_gpu is None else f'controller-gpu{args.single_gpu}.lock'
 with (OUT/lock_name).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with transaction() as s:
   assert s['notebook_id'].split('-')[-1] in socket.gethostname()
   s.update(hostname=socket.gethostname(),controller_pid=os.getpid(),controller_started_utc=now())
  with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(slot,gpus))
  subprocess.run(['python',str(ROOT/'scripts/libero_4rc_a1of4_pro_eval/report.py')],check=False)
  print('QUEUE_FINISHED; independent idle GPU guard remains active',flush=True)
if __name__=='__main__':main()
