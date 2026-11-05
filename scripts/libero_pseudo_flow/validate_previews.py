import json,time
from pathlib import Path
import av,numpy as np
from vpp2.libero_pseudo_flow.geometry import read_record,normalize_clip,atomic_json
root=Path('runs/libero_pseudo_flow_4rc_20261102')
assert json.loads((root/'geometry-tests.json').read_text())['passed']
assert json.loads((root/'checkpoint-header.json').read_text())['passed']
reports=[]
for suite in ('spatial','object','goal','long'):
 p=root/'previews'/suite;d=p/'episode_000000';m=json.loads((d/'meta.json').read_text());parity=json.loads((d/'decoder_parity.json').read_text());stats=json.loads((p/'4rc_dt4.json').read_text());comparison=json.loads((p/'gt_vs_4rc_dt4.json').read_text())
 assert m['frames']==64 and not m['uses_gt_labels'] and not m['uses_actions']
 assert parity['passed'] and parity['max_abs']<.0005
 assert abs(stats['normalized_mean_radius']-1)<1e-5
 assert max(map(abs,stats['normalized_point_mean']))<1e-4
 videos={}
 for name,count,width in [('4rc_dt4.mp4',60,448),('gt_vs_4rc_dt4.mp4',60,672),('4rc_tracks.mp4',64,448)]:
  with av.open(str(p/name)) as c:
   decoded=list(c.decode(video=0));assert len(decoded)==count and all(f.width==width and f.height==224 for f in decoded)
  videos[name]=dict(frames=count,width=width,height=224,fps=20)
 for i in (0,16,32,48,63):
  r=read_record(d/f'frame_{i:06d}.npz.lz4');assert r['point_map'].shape==(224,224,3) and r['point_map'].dtype==np.float32
  for dt in (1,4):
   assert np.isfinite(r[f'scene_flow_dt{dt}']).all()
   if i+dt>=64:assert not r[f'flow_valid_mask_dt{dt}'].any() and not r[f'scene_flow_dt{dt}'].any()
 reports.append(dict(suite=suite,parity=parity,normalization=stats,gt_comparison=comparison,videos=videos,window_alignment=m['windows']))
result=dict(passed=True,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),visual_inspection='Assistant inspected rendered contact sheets from all four suites; source-pixel alignment and motion directions coherent',limitations='Small sample only; depth errors and nonzero background pseudo motion remain; no GT calibration used',samples=reports)
atomic_json(root/'preview-validation.json',result)
print('PASS: four suites, real weights, full reference decoder parity, normalization, LZ4 tensors and all 12 decoded videos.')
