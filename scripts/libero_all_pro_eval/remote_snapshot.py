import json,os,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103'
sys.path.insert(0,str(ROOT.parent/'tools/qzcli_tool'))
from qzcli.cli import _exec_launch
for k in list(os.environ):
 if k.lower().endswith('_proxy'):os.environ.pop(k,None)
class Quiet:
 def print(self,*a,**kw):pass
 def print_error(self,*a,**kw):print('remote snapshot error')
 def print_success(self,*a,**kw):pass
 def print_warning(self,*a,**kw):pass
code='''import subprocess,json,time,socket
from pathlib import Path
def run(args):
 p=subprocess.run(args,text=True,capture_output=True);return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
d=dict(time=time.time(),hostname=socket.gethostname(),gpus=run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader']),compute=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),tmux=run(['/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/tools/tmux-portable/tmux','list-sessions']))
p=run(['ps','-eww','-o','pid,ppid,etime,args'])
d['evaluation_processes']=[l for l in p['stdout'].splitlines() if any(k in l for k in ['-m vpp2.libero','controller.py','gpu_idle_guard.py','libero_all_pro_eval/preflight.py','libero_all_pro_eval/validate.py']) and 'python3 -c' not in l]
Path(DEST).write_text(json.dumps(d,indent=2)+'\\n')
'''.replace('DEST',repr(str(OUT/'remote-snapshot.json')))
info=json.loads((OUT/'jupyter-private.json').read_text());job=_exec_launch(info,'python3 -c '+shlex.quote(code),Quiet());assert job;print('snapshot requested')
