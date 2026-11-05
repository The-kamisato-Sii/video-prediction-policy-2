"""Actual-GT equivalence, cache failure handling, and worker isolation checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .motion_records import MotionRecords, cache_path
from .motion_reader import MotionReader
from .runtime import load_config, create_dataset
from .data import collate_flow_action
from vpp2.libero.motion_data import load_motion_clip


def same_clip(a,b):
    for camera in a['cameras']:
        for key in a['cameras'][camera]:
            assert np.array_equal(a['cameras'][camera][key],b['cameras'][camera][key]),(camera,key)
    for key in ('frame_index','requested_frame_index','flow_target_frame_index','center'):
        assert np.array_equal(a[key],b[key]),key
    for key in ('scale','sample_stride','flow_dt','prediction_horizon','normalized'):
        assert a[key]==b[key],key


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    report=dict(normalized_checks=[],cache_failures={},loader={})
    reader=MotionReader('records')
    for source in a.source:
        with MotionRecords(cache_path(source)) as f:n=f.n
        for start in sorted({0,1,2,3,17,n-33}):
            if start+32>=n:continue
            for dt in (1,4):
                ref=load_motion_clip(source,start=start,length=8,stride=4,flow_dt=dt,prediction_horizon=32)
                fast=reader.load(source,start=start,flow_dt=dt)
                same_clip(ref,fast)
                report['normalized_checks'].append(dict(source=source,start=start,dt=dt,exact_equal=True))
        print('normalized_clip_checks_passed',source,flush=True)
    reader.close()
    # Mutate ONLY owned temporary copies; real GT and real sidecars stay intact.
    with tempfile.TemporaryDirectory(prefix='vpp2_motion_integrity_') as root:
        original_source=Path(a.source[0]).resolve()
        source=Path(root)/original_source.parents[2].name/'motions/chunk-000'/original_source.name
        source.parent.mkdir(parents=True)
        shutil.copy2(a.source[0],source)
        meta=source.parents[2]/'meta/motions'/(source.stem+'.json')
        meta.parent.mkdir(parents=True)
        shutil.copy2(original_source.parents[2]/'meta/motions'/(source.stem+'.json'),meta)
        target=cache_path(source);target.parent.mkdir(parents=True)
        ref=load_motion_clip(source,start=0,length=8,stride=4,flow_dt=4,prediction_horizon=32)
        auto=MotionReader('auto')
        same_clip(ref,auto.load(source));assert auto.last_backend=='hdf5'
        report['cache_failures']['missing_falls_back']=True
        shutil.copy2(cache_path(a.source[0]),target)
        same_clip(ref,auto.load(source));assert auto.last_backend=='records'
        report['cache_failures']['appearing_cache_picked_up']=True
        auto.close()
        with MotionRecords(target) as f:pos=int(f.offsets[0])+40
        with target.open('r+b') as f:
            f.seek(pos);b=f.read(1);f.seek(pos);f.write(bytes([b[0]^0x80]))
        with warnings.catch_warnings(record=True) as logged:
            same_clip(ref,auto.load(source));assert auto.last_backend=='hdf5'
            assert logged
        report['cache_failures']['corruption_falls_back_with_warning']=True
        strict=MotionReader('records')
        try:strict.load(source)
        except (ValueError,RuntimeError,OSError):report['cache_failures']['strict_corruption_rejected']=True
        else:raise AssertionError('Corruption was not detected')
        strict.close();auto.close()
        auto=MotionReader('auto')  # independent fault, independent warn-once state
        shutil.copy2(cache_path(a.source[0]),target)
        st=source.stat();os.utime(source,ns=(st.st_atime_ns,st.st_mtime_ns+1000000000))
        with warnings.catch_warnings(record=True) as logged:
            same_clip(ref,auto.load(source));assert auto.last_backend=='hdf5' and logged
        report['cache_failures']['stale_source_rejected']=True
        auto.close()
    print('cache_failure_checks_passed',flush=True)
    cfg=load_config(['flow.action_denominator=8','flow.missing_motion=skip','flow.motion_backend=records',
        'flow.available_manifest=runs/libero_flow_manifests/flow-ddp-existing-h200-1023-r2.json'])
    ds=create_dataset(cfg)
    chosen={str(Path(s).resolve()) for s in a.source}
    indices=[]
    for i,r in enumerate(ds.episodes):
        if r['indicator'] and str(Path(r['motion']).resolve()) in chosen:
            offset=ds.ends[i-1] if i else 0
            indices.extend(offset+x for x in (0,1,2,3,r['starts']-1))
    assert len(indices)==len(a.source)*5
    with patch.object(ds.motion_reader,'load',side_effect=AssertionError('Action accessed motion')):
        action=ds[ds.first_index(0)];assert 'motion' not in action
    # Flow branch must continue to exclude the action column at the parquet read.
    import pyarrow.parquet as pq
    original=pq.read_table
    def guarded(*args,**kwargs):
        assert 'action' not in kwargs['columns'];return original(*args,**kwargs)
    with patch('vpp2.libero_flow.data.pq.read_table',side_effect=guarded):
        ds._tables.clear();sample=ds[indices[0]];assert 'action' not in sample
    cfg.flow.motion_backend='hdf5';legacy=create_dataset(cfg)
    for index in indices:
        before,after=legacy[index],ds[index]
        for key in before:
            if torch.is_tensor(before[key]):assert torch.equal(before[key],after[key]),key
        assert after['provenance']['motion_storage']=='records'
    report['dataset_samples_bitwise_equal']=len(indices)
    # Reader was opened in the parent above. Fork workers must reopen its files.
    sample_indices=indices+[ds.first_index(0)]*4
    loader=DataLoader(Subset(ds,sample_indices),batch_size=4,num_workers=2,
        collate_fn=collate_flow_action,prefetch_factor=1)
    total=flow=action_count=0
    for batch in loader:
        total+=len(batch['indicator']);flow+=len(batch['flow_indices']);action_count+=len(batch['action_indices'])
        for provenance in batch['provenance']:
            if provenance['motion_opened']:assert provenance['motion_storage']=='records'
        if 'motion' in batch:assert tuple(batch['motion'].shape[1:])==(len(ds.motion_cameras),6,8,224,224)
        if 'action' in batch:assert tuple(batch['action'].shape[1:])==(32,7)
    assert total==len(sample_indices) and flow==len(indices) and action_count==4
    report['loader']=dict(workers=2,samples=total,flow=flow,action=action_count,all_flow_used_records=True,
        missing_label_access_guards_passed=True,parent_open_before_fork=True)
    ds.motion_reader.close();legacy.motion_reader.close()
    report['status']='passed'
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
