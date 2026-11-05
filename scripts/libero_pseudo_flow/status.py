"""Summarize completed pseudo episodes and live worker progress without GPUs."""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
repo=Path(__file__).resolve().parents[2]
run=repo/'runs/libero_pseudo_flow_4rc_20261102'
manifest=json.loads((run/'input-manifest.json').read_text())
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
rows=[]
for suite,source in manifest['suites'].items():
 root=repo/Path(source['dataset']).relative_to('video-prediction-policy-2')/'pseudo_flow/4rc_v1'
 with ThreadPoolExecutor(8) as ex:metas=list(ex.map(read,root.glob('episode_*/meta.json')))
 complete=[m for m in metas if m.get('complete_episode')]
 workers=[]
 for gpu in (0,1):
  state=read(run/f'production/{suite}/gpu{gpu}.json')
  if state.get('state')=='RUNNING':
   progress=read(root/f"episode_{state['episode']:06d}/progress.json")
   state['progress']=progress
  workers.append(dict(gpu=gpu,**state))
 rows.append(dict(suite=suite,completed_episodes=len(complete),expected_episodes=source['episodes'],completed_frames=sum(m['frames'] for m in complete),expected_frames=source['frames'],workers=workers))
report=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),suites=rows)
(run/'status-summary.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:
 print(r['suite'],f"{r['completed_episodes']}/{r['expected_episodes']} episodes",f"{r['completed_frames']}/{r['expected_frames']} frames",[(w['gpu'],w.get('state','NOT_STARTED'),w.get('episode')) for w in r['workers']])
