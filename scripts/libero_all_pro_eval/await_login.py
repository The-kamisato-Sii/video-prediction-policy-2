"""Start the already authorized evaluation once the pending device login completes."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103'
CLI=ROOT.parent/'tools/inspire-cli/bin/qz';PYTHON=ROOT.parent/'tools/qzcli-env/bin/python'
os.environ['QZ_CLI_CONFIG_DIR']=str(ROOT.parent/'tools/inspire-cli/state')
def state(status,**kw):
 p=OUT/'startup-status.json';p.write_text(json.dumps(dict(status=status,time=time.time(),**kw),indent=2)+'\n')
with (OUT/'auth-watcher.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state('WAITING_FOR_PLATFORM_LOGIN');deadline=time.time()+3600
 while time.time()<deadline:
  p=subprocess.run([str(CLI),'workspace','ListWorkspaces','--page-number','1','--page-size','100'],capture_output=True,text=True)
  if p.returncode==0:
   with (OUT/'startup.log').open('a') as log:
    result=subprocess.run([str(PYTHON),str(ROOT/'scripts/libero_all_pro_eval/start_remote.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:state('REMOTE_START_FAILED',see='startup.log');raise SystemExit(result.returncode)
   state('REMOTE_LAUNCH_SENT');break
  time.sleep(10)
 else:state('LOGIN_WAIT_EXPIRED');raise SystemExit(1)
 # Record evidence from the remote launcher without accessing credentials.
 deadline=time.time()+1800
 while time.time()<deadline:
  if (OUT/'queue.log').exists():
   s=json.loads((OUT/'queue.json').read_text());running=sum(j['state']=='RUNNING' for j in s['jobs'].values());complete=sum(j['state']=='COMPLETE' for j in s['jobs'].values())
   if running or complete:state('EVALUATION_RUNNING',running_jobs=running,completed_jobs=complete);break
  if (OUT/'preflight.log').exists() and 'Traceback' in (OUT/'preflight.log').read_text():state('PREFLIGHT_FAILED',see='preflight.log');break
  if (OUT/'validation.log').exists() and 'Traceback' in (OUT/'validation.log').read_text():state('CHECKPOINT_CHECK_FAILED',see='validation.log');break
  time.sleep(10)
