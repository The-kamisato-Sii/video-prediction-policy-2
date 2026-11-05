"""Regression audit: head selection leaves RGB/action/normalization unchanged."""
from pathlib import Path
import json, gc
from unittest.mock import patch
import torch
from vpp2.libero_flow.runtime import load_config,create_dataset
from vpp2.libero_flow.data import collate_flow_action
from vpp2.libero_flow.views import check_checkpoint_views
from vpp2.utils import misc
out=Path('runs/libero_flow_head_only_20261025');out.mkdir(exist_ok=True)
misc.register_work_dir(out)
cfg=load_config(['wandb.enabled=false','flow.available_manifest=runs/libero_flow_submit_20261025/flow_manifest.json'])
assert list(cfg.flow.motion_cameras)==['image']
checks=[]
for den in [8,4,2]:
 cfg.flow.action_denominator=den;ds=create_dataset(cfg)
 assert ds.motion_cameras==('image',) and ds.audit['flow_episodes']==1589
 assert ds.audit['action_episodes']==1712//den
 checks.append(dict(denominator=den,audit=ds.audit))
 if den==8:
  for suite in ['spatial','object','goal','libero_10']:
   idx=ds.first_index(1,suite)
   for dt in [1,4]:
    ds.flow_dt=dt
    torch.manual_seed(42);head=ds[idx]
    ds.motion_cameras=('image','wrist_image')
    torch.manual_seed(42);dual=ds[idx]
    ds.motion_cameras=('image',)
    assert head['motion'].shape==(1,6,8,224,224)
    for k in ['motion','motion_valid']:assert torch.equal(head[k],dual[k][:1]),k
    for k in ['video','proprio','context','motion_center','motion_scale']:assert torch.equal(head[k],dual[k]),k
    assert head['video'].shape[-2:]==(224,448)
    assert head['motion_frame_index'].tolist()==list(range(0,32,4))
    assert head['flow_target_frame_index'].tolist()==[i+dt for i in range(0,32,4)]
    assert 'action' not in head
    checks.append(dict(suite=suite,dt=dt,gt_shape=list(head['motion'].shape),rgb_shape=list(head['video'].shape),head_equals_dual_first_view=True))
  with patch.object(ds.motion_reader,'load',side_effect=AssertionError('action must not read flow')):
   ds.motion_cameras=('image',);torch.manual_seed(42);a=ds[ds.first_index(0)]
   ds.motion_cameras=('image','wrist_image');torch.manual_seed(42);a2=ds[ds.first_index(0)]
   for k in ['action','video','proprio','context']:assert torch.equal(a[k],a2[k]),k
  assert 'motion' not in a
  torch.save(collate_flow_action([head]),out/'head_flow_batch.pt')
  torch.save(collate_flow_action([dual]),out/'dual_flow_batch.pt')
  torch.save(collate_flow_action([a]),out/'action_batch.pt')
 del ds;gc.collect()
check_checkpoint_views({'motion_cameras':['image']},('image',))
for payload in [{},{'motion_cameras':['image','wrist_image']}]:
 try:check_checkpoint_views(payload,('image',))
 except ValueError:pass
 else:raise AssertionError('Dual-view resume was not rejected')
(out/'data_validation.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2))
print('PASS: all 4 suites, dt=1/4, fractions=1/8,1/4,1/2; action/RGB/stats unchanged; no hidden action labels; incompatible resume rejected',flush=True)
