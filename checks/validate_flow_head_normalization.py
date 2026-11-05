"""Selected-camera normalization: upstream parity, wrist isolation and real loaders."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from vpp2.libero_flow.motion_records import MotionRecords,cache_path,phase_position,normalize_clip
from vpp2.libero_flow.motion_reader import MotionReader
from vpp2.libero_flow.runtime import load_config,create_dataset
from vpp2.libero_flow.data import collate_flow_action
from vpp2.libero_flow.codec import MotionCodec
from vpp2.libero_flow.model import FlowActionAdapters

OUT=Path('runs/libero_flow_headnorm_20261026');OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
def load_module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
up=load_module('runs/motioncrafter_audit_20261026/official/datasets/video_transforms.py','upstream_headnorm')
legacy=load_module(OUT/'before/src/vpp2/libero_flow/data.py','vpp2.libero_flow.data_before_headnorm')
normalizer=up.WorldMapNormalize()
report=dict(clip_checks=[],loader_checks=[])
fast=MotionReader('records',normalization_cameras=['image'])
slow=MotionReader('hdf5',normalization_cameras=['image'])
for suite in ('spatial','object','goal','10'):
 source=Path(f'data/libero/libero_{suite}_no_noops_lerobot/motions/chunk-000/episode_000000.hdf5')
 with MotionRecords(cache_path(source)) as r:
  for start in (0,17,r.n-33):
   raw,masks=r.read_rows(phase_position(start,r.n),8)
   for dt in (1,4):
    got=fast.load(source,start=start,flow_dt=dt)
    ref=slow.load(source,start=start,flow_dt=dt)
    assert got['normalization_cameras']==['image']
    assert np.array_equal(got['center'],ref['center']) and got['scale']==ref['scale']
    for key in ('point_map','scene_flow','valid_mask','flow_valid_mask'):
     assert np.array_equal(got['cameras']['image'][key],ref['cameras']['image'][key]),(suite,start,dt,key)
    points=torch.from_numpy(raw[:,0,0].copy());valid=torch.from_numpy(masks[:,0,0].copy())
    flow=torch.from_numpy(raw[:,0,1 if dt==1 else 2].copy())
    p,_,f,info=normalizer.normalize(points,valid,torch.eye(4).repeat(8,1,1),flow)
    actual=got['cameras']['image'];fv=actual['flow_valid_mask']
    pdiff=float(np.abs(p.numpy()[valid]-actual['point_map'][valid]).max())
    fdiff=float(np.abs(f.numpy()[fv]-actual['scene_flow'][fv]).max())
    assert pdiff<3e-6 and fdiff<3e-6,(pdiff,fdiff)
    xy=actual['point_map'][actual['valid_mask']].astype(np.float64)
    center_error=float(np.linalg.norm(xy.mean(0)));radius=float(np.linalg.norm(xy,axis=-1).mean())
    assert center_error<1e-6 and abs(radius-1)<1e-6
    changed=raw.copy();changed[:,1]=changed[:,1]*97+123
    changed_masks=masks.copy();changed_masks[:,1]=False
    perturbed=normalize_clip(changed,changed_masks,start,r.n,dt,normalization_cameras=['image'])
    assert np.array_equal(got['center'],perturbed['center']) and got['scale']==perturbed['scale']
    assert np.array_equal(actual['point_map'],perturbed['cameras']['image']['point_map'])
    assert np.array_equal(actual['scene_flow'],perturbed['cameras']['image']['scene_flow'])
    report['clip_checks'].append(dict(suite=suite,start=start,dt=dt,hdf5_records_exact=True,
      official_point_max_abs=pdiff,official_flow_max_abs=fdiff,normalized_center_norm=center_error,
      normalized_mean_radius=radius,wrist_values_and_masks_no_effect=True))
 print('clip checks passed',suite,flush=True)
fast.close();slow.close()
sample=None
for den in (8,4,2):
 cfg=load_config([f'flow.action_denominator={den}',
  'flow.available_manifest=runs/libero_flow_submit_20261025/flow_manifest.json'])
 ds=create_dataset(cfg)
 assert ds.audit['motion_normalization_cameras']==['image'] and ds.audit['flow_episodes']==1589
 for suite in ('spatial','object','goal','10'):
  flow_idx=ds.first_index(1,suite=suite);action_idx=ds.first_index(0,suite=suite)
  flow=ds[flow_idx]
  assert flow['provenance']['motion_normalization_cameras']==['image']
  assert flow['provenance']['motion_storage']=='records'
  assert flow['motion'].shape==(1,6,8,224,224) and torch.isfinite(flow['motion']).all()
  points=flow['motion'][0,:3].permute(1,2,3,0)[flow['motion_valid'][0,0]].double()
  assert points.mean(0).norm()<1e-6
  assert abs(points.norm(dim=-1).mean()-1)<1e-6
  with patch.object(ds.motion_reader,'load',side_effect=AssertionError('action must not read flow')):
   action=ds[action_idx];old_action=legacy.FlowActionDataset.__getitem__(ds,action_idx)
  for key,value in action.items():
   if torch.is_tensor(value):assert torch.equal(value,old_action[key]),key
  report['loader_checks'].append(dict(denominator=den,suite=suite,flow_shape=list(flow['motion'].shape),
    flow_backend='records',normalization_cameras=['image'],action_unchanged=True))
  if sample is None:sample=collate_flow_action([flow])
 report[f'dataset_audit_1of{den}']=ds.audit
 ds.motion_reader.close()
 print('loader checks passed',den,flush=True)
# Confirm the new labels still pass through the unchanged frozen codec and token adapter.
with torch.no_grad():
 codec=MotionCodec('weights/MotionCrafter/geometry_motion_vae',cameras=['image']).cuda().bfloat16()
 maps=sample['motion'].cuda().bfloat16();z=codec.encode(maps);decoded=codec.decode(z)
 adapter=FlowActionAdapters(tokens=32,motion_channels=8).cuda().bfloat16()
 tokens=adapter.embed_motion(z);output=adapter.output(tokens,1)
 assert z.shape==(1,8,8,28,28) and tokens.shape==(1,32,512) and output.shape==z.shape
 assert decoded.shape==maps.shape and torch.isfinite(z).all() and torch.isfinite(decoded).all()
 report['codec']=dict(motion=list(maps.shape),latent=list(z.shape),tokens=list(tokens.shape),decoded=list(decoded.shape),scaling_factor=codec.scaling_factor,finite=True)
report['status']='passed'
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('ALL_HEAD_NORMALIZATION_CHECKS_PASSED',flush=True)
