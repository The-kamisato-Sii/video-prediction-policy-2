import json,struct
from pathlib import Path
import torch
from arc.models.arc.arc import Arc
from vpp2.libero_pseudo_flow.geometry import atomic_json
torch.set_num_threads(4)
p=Path('weights/4RC/download_parts/0000.tmp')
if not p.exists():p=p.with_suffix('')
with p.open('rb') as f:
 size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
model=Arc();state=model.state_dict();keys=set(header)-{'__metadata__'}
assert keys==set(state),(keys-set(state),set(state)-keys)
for k,t in state.items():assert list(t.shape)==header[k]['shape'],k
report=dict(passed=True,tensors=len(state),parameters=sum(p.numel() for p in model.parameters()),strict_keys_and_shapes_match=True,checkpoint_header_bytes=size,full_weights_loaded=False)
atomic_json('runs/libero_pseudo_flow_4rc_20261102/checkpoint-header.json',report)
print(report,flush=True)
