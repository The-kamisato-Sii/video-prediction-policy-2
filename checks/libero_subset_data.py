"""Validate LIBERO episode subsets against exact full-data samples on CPU.

Run from the repository root with the vpp2 environment and PYTHONPATH=src.
Requires the four local LeRobot datasets and their cached text embeddings.
Does not instantiate a policy or allocate GPUs.
"""
import argparse, copy, json, os
from pathlib import Path
os.environ['VPP2_VIDEO_DECODE_UINT8']='1'
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from vpp2.utils import misc
from vpp2.datasets.lerobot.episode_selection import load_episode_selection
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', default='runs/libero_subset_validation')
args=parser.parse_args()
out=Path(args.output_dir)
out.mkdir(parents=True,exist_ok=True)
misc.register_work_dir(out)
cfg=OmegaConf.load('configs/libero/train_action.yaml').data.train
print('Building unchanged full dataset',flush=True)
full=instantiate(cfg)
assert len(full)==277713
result=[]; previous=None
for divisor,n,frames in [(8,214,34900),(4,428,69942),(2,856,139789)]:
    path=Path(f'configs/libero/subsets/episodes_1of{divisor}_seed42.json')
    mapping,audit,_=load_episode_selection(path,cfg.dataset_dirs)
    if previous:
        assert all(set(previous[k]) <= set(mapping[k]) for k in mapping)
    previous=mapping
    selected_cfg=copy.deepcopy(cfg)
    selected_cfg.episode_manifest=str(path)
    selected_cfg.episode_manifest_sha256=audit['sha256']
    selected=instantiate(selected_cfg)
    assert len(selected)==frames and audit['episodes']==n
    full_offset=selected_offset=0
    checked=0;tasks=set()
    for ds_idx,(orig,sub) in enumerate(zip(full.lerobot_dataset.multi_dataset._datasets,selected.lerobot_dataset.multi_dataset._datasets)):
        ids=set(mapping[str(cfg.dataset_dirs[ds_idx])])
        raw=sub.hf_dataset.with_format(None)
        assert set(raw['episode_index'])==ids
        assert set(sub.episodes)==ids
        assert len(raw)==sum(sub.meta.episodes[i]['length'] for i in ids)
        tasks.update((ds_idx,int(t)) for t in raw['task_index'])
        # Check both boundaries of the first and last retained episodes against
        # the exact same original frames, including end-of-episode padding.
        for local_ep in (0,len(sub.episodes)-1):
            ep=sub.episodes[local_ep]
            for delta in (0,int(sub.meta.episodes[ep]['length'])-1):
                sub_idx=selected_offset+int(sub.episode_data_index['from'][local_ep])+delta
                full_idx=full_offset+int(orig.episode_data_index['from'][ep])+delta
                _,a=selected.get_strict_item(sub_idx)
                _,b=full.get_strict_item(full_idx)
                assert a.keys()==b.keys()
                for key in a:
                    if isinstance(a[key],torch.Tensor):
                        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                    else:
                        assert a[key]==b[key],key
                checked+=1
        full_offset+=len(orig);selected_offset+=len(sub)
    assert len(tasks)==40
    # Force an actual Base loader retry after an I/O error. The replacement
    # index must refer to the selected parquet rows, never the full dataset.
    base=selected.lerobot_dataset
    original_multi=base.multi_dataset
    class FailOnce:
        num_frames=original_multi.num_frames
        calls=[]
        def __getitem__(self,idx):
            self.calls.append(idx)
            assert 0<=idx<len(selected)
            if len(self.calls)==1:
                raise OSError('intentional retry test')
            return original_multi[idx]
    fail=FailOnce();base.multi_dataset=fail
    try:
        actual,_=base._load_lerobot_sample(0,allow_random_fallback=True)
        assert len(fail.calls)==2 and actual==fail.calls[-1]
    finally:
        base.multi_dataset=original_multi
    result.append(dict(**audit,matched_boundary_samples=checked,tasks=40,retry='PASS'))
    print(json.dumps(result[-1]),flush=True)
(out/'real-data-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: all subsets, exact baseline sample equality, nested episodes, all 40 tasks, bounded retries',flush=True)
