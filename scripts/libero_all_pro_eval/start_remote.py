"""Find the user's named 8-GPU notebook and start the prepared evaluation."""
import json,os,re,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103';HOME_ROOT=ROOT.parent
CLI=HOME_ROOT/'tools/inspire-cli/bin/qz'
def call(args):
 p=subprocess.run([str(CLI),*args],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr.strip() or 'Inspire CLI request failed')
 d=json.loads(p.stdout)
 if d.get('ResponseMetadata',{}).get('Error'):raise RuntimeError(str(d['ResponseMetadata']['Error']))
 return d['Result']
def records(result):
 if isinstance(result,list):return result
 for k in ('list','workspaces','items','data'):
  if isinstance(result.get(k),list):return result[k]
 raise ValueError('Unexpected list response structure')
def main():
 workspaces=records(call(['workspace','ListWorkspaces','--page-number','1','--page-size','100']))
 matches=[]
 for ws in workspaces:
  wid=ws.get('workspace_id') or ws.get('id')
  if not wid:continue
  for n in records(call(['notebook','ListNotebooks','--workspace-id',wid,'--scope','TASK_LIST_SCOPE_MINE','--page-number','1','--page-size','100'])):
   normalized=re.sub('[^a-z0-9]','',n.get('name','').lower())
   if normalized.startswith('liberoprotest') and n.get('status')=='RUNNING':matches.append((wid,n))
 if len(matches)!=1:raise RuntimeError(f'Expected one running libero pro test notebook, found {len(matches)}')
 wid,n=matches[0];nid=n['notebook_id']
 detail=call(['notebook','GetNotebook','--notebook-id',nid])
 (OUT/'notebook.json').write_text(json.dumps(dict(notebook_id=nid,name=n['name'],workspace_id=wid,status=detail.get('status')),indent=2)+'\n')
 s=json.loads((OUT/'queue.json').read_text())
 if s.get('notebook_id') not in (None,nid):raise RuntimeError('Queue belongs to another notebook')
 s['notebook_id']=nid;s['notebook_name']=n['name'];(OUT/'queue.json').write_text(json.dumps(s,indent=2)+'\n')
 urls=call(['notebook','GetNotebookAccessUrl','--notebook-id',nid])
 match=re.search(r'(https://[^/]+/[^/]+/[^/]+/[^/]+/jupyter/[^/]+/([^/]+))/lab',urls['jupyter_url'])
 if not match:raise ValueError('Unrecognized Jupyter access URL')
 info=dict(base_url=match.group(1),token=match.group(2),notebook_id=nid)
 private=OUT/'jupyter-private.json';private.write_text(json.dumps(info));private.chmod(0o600)
 sys.path.insert(0,str(HOME_ROOT/'tools/qzcli_tool'))
 from qzcli.cli import _exec_launch
 for k in list(os.environ):
  if k.lower().endswith('_proxy'):os.environ.pop(k,None)
 class Quiet:
  def print(self,*a,**kw):pass
  def print_error(self,*a,**kw):print('Remote execution failed; private details suppressed')
  def print_success(self,*a,**kw):pass
  def print_warning(self,*a,**kw):pass
 command='bash '+shlex.quote(str(ROOT/'scripts/libero_all_pro_eval/launch.sh'))+' > '+shlex.quote(str(OUT/'node-launcher.log'))+' 2>&1'
 job=_exec_launch(info,command,Quiet())
 if not job:raise RuntimeError('Could not launch remote evaluation')
 result=dict(state='REMOTE_LAUNCH_SENT',notebook_id=nid,notebook_name=n['name'],exec_job=job,time=time.time(),expected_gpus=8,rollouts=6000)
 (OUT/'remote-launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
