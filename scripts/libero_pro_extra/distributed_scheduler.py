"""Four GPU work stealing queue, waiting for the existing evaluations per slot."""
import argparse,datetime,fcntl,json,os,socket,subprocess,time
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
STORE=OUT/'queue.json';OLD=ROOT/'runs/libero_flow_eval_20261030/distributed-queue.json'
LAUNCH=ROOT/'scripts/libero_pro_extra/run_job.sh'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextmanager
def transaction():
    with (OUT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);s=json.loads(STORE.read_text());yield s
        s['updated_utc']=now();p=OUT/f'.queue.{os.getpid()}.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STORE)

def old_busy(node,gpu):
    s=json.loads(OLD.read_text())
    # Original scheduler still owns the next available slot until its READY jobs are claimed.
    return any(j['state']=='READY' or (j['state']=='RUNNING' and j.get('node')==node and j.get('gpu')==gpu) for j in s['models'].values())

def claim(node,gpu):
    if old_busy(node,gpu):return None
    with transaction() as s:
        for key,j in s['jobs'].items():
            if j['state']=='RUNNING' and j.get('node')==node and j.get('gpu')==gpu:return key,dict(j),True
        for key,j in s['jobs'].items():
            if j['state'] not in ('WAIT_ASSETS','WAIT_CHECKPOINT','READY'):continue
            m=s['models'][j['model']]
            if not Path(m['config']['action_checkpoint']).is_file():j['state']='WAIT_CHECKPOINT';continue
            gate=OUT/'asset_checks'/f"{j['axis']}.json"
            if not gate.exists() or json.loads(gate.read_text()).get('status')!='PASS':j['state']='WAIT_ASSETS';continue
            # A process interrupted after writing RUN_CONFIG must keep its GPU id.
            contract=OUT/'jobs'/key/'evaluation/RUN_CONFIG.json'
            if contract.exists() and json.loads(contract.read_text())['gpus']!=[gpu]:continue
            j.update(state='RUNNING',node=node,gpu=gpu,pid=None,started_utc=now())
            return key,dict(j),False
    return None

def progress(key,node,gpu):
    count=dict(episodes=0,successes=0)
    for p in (OUT/'jobs'/key/'evaluation/results').rglob('*_results.json'):
        try:
            r=json.loads(p.read_text());count['episodes']+=len(r['success_episodes'])+len(r['failure_episodes']);count['successes']+=len(r['success_episodes'])
        except (ValueError,KeyError,OSError):continue
    with transaction() as s:s['jobs'][key].update(count,last_heartbeat_utc=now());s['nodes'][node]['heartbeat_utc']=now()

def alive(pid,key):
    if not pid:return False
    try:
        p=Path(f'/proc/{pid}');stat=(p/'stat').read_text()
        if stat[stat.rfind(')')+2:].split()[0]=='Z':return False
        args=(p/'cmdline').read_bytes().split(b'\0');return str(LAUNCH).encode() in args and key.encode() in args
    except FileNotFoundError:return False

def final_audit(key,gpu):
    f=OUT/'jobs'/key;c=json.loads((f/'eval.yaml').read_text());r=json.loads((f/'evaluation/summary.json').read_text())
    assert r['complete'] and r['episodes']==400 and not r['issues'] and not r['missing']
    assert (f/'evaluation/full_run.status').read_text().strip()=='0'
    assert r['action_nfe']==c['action_nfe'] and r['extra_axis']==c['extra_axis']
    if c['extra_model_kind']=='flow':
        p=json.loads((f/f'inference-parity-gpu{gpu}.json').read_text())
        assert p['status']=='PASS' and p['max_abs_vs_training_sample_task']==0 and p['all_saved_tensors_equal']
    return r

def slot(node,gpu):
    while True:
        item=claim(node,gpu)
        if item is None:
            with transaction() as s:
                s['nodes'][node]['heartbeat_utc']=now()
                pending=any(x['state'] in ('RUNNING','WAIT_ASSETS','WAIT_CHECKPOINT','READY') for x in s['jobs'].values())
            if not pending:return
            time.sleep(15);continue
        key,j,adopt=item;folder=OUT/'jobs'/key
        try:
            if adopt:
                pid=j.get('pid')
                while alive(pid,key):progress(key,node,gpu);time.sleep(15)
                code=int((folder/'exit.status').read_text().strip())
            else:
                with (folder/'coordinator.log').open('a') as log:
                    log.write(f'\nSTART {now()} {node} GPU{gpu}\n');log.flush()
                    p=subprocess.Popen(['bash',str(LAUNCH),key,str(gpu)],stdout=log,stderr=subprocess.STDOUT)
                    with transaction() as s:s['jobs'][key]['pid']=p.pid
                    print(json.dumps(dict(event='START',job=key,node=node,gpu=gpu,pid=p.pid)),flush=True)
                    while p.poll() is None:progress(key,node,gpu);time.sleep(15)
                    code=p.returncode
            progress(key,node,gpu)
            if code!=0:raise RuntimeError(f'Launcher exit {code}')
            r=final_audit(key,gpu)
            with transaction() as s:s['jobs'][key].update(state='COMPLETE',ended_utc=now(),episodes=400,successes=r['successes'],success_rate=r['success_rate'])
            print(json.dumps(dict(event='COMPLETE',job=key,node=node,gpu=gpu)),flush=True)
        except Exception as e:
            with transaction() as s:s['jobs'][key].update(state='FAILED',ended_utc=now(),error=repr(e))
            print(json.dumps(dict(event='FAILED',job=key,error=repr(e))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['h200-a','h200-b'],required=True);a=p.parse_args()
    d=OUT/'nodes'/a.node;d.mkdir(parents=True,exist_ok=True)
    with (d/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with transaction() as s:
            assert s['nodes'][a.node]['notebook_id'].split('-')[-1] in socket.gethostname()
            s['nodes'][a.node].update(pid=os.getpid(),hostname=socket.gethostname(),heartbeat_utc=now())
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda g:slot(a.node,g),(0,1)))
if __name__=='__main__':main()
