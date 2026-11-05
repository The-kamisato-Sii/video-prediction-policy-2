"""Real CUDA codec/adapters checks for one-view supervision and all token grids."""
from pathlib import Path
from unittest.mock import patch
import json,gc
import torch
from vpp2.libero_flow.codec import MotionCodec
from vpp2.libero_flow.reconstruction_codec import ReconstructionCodec
from vpp2.libero_flow.model import FlowActionAdapters
out=Path('runs/libero_flow_head_only_20261025');report={};dtype=torch.bfloat16
head=torch.load(out/'head_flow_batch.pt',weights_only=True)['motion'].cuda().to(dtype)
dual=torch.load(out/'dual_flow_batch.pt',weights_only=True)['motion'].cuda().to(dtype)
c=MotionCodec('weights/MotionCrafter/geometry_motion_vae',cameras=['image','wrist_image']).cuda().to(dtype)
z2=c.encode(dual)
with torch.no_grad():
 assert c.decode(z2).shape==dual.shape
c.motion_cameras=('image',);c.num_views=1;c.latent_channels=8
z=c.encode(head)
assert torch.equal(z,z2[:,:8]);assert z.shape==(1,8,8,28,28)
with torch.no_grad():decoded=c.decode(z)
assert decoded.shape==head.shape and torch.isfinite(decoded).all()
report['frozen_codec']=dict(latent=list(z.shape),decoded=list(decoded.shape),head_latent_equals_old_first_view=True,frozen=not any(p.requires_grad for p in c.parameters()))
rows=[]
for n in [32,64,128]:
 a=FlowActionAdapters(tokens=n).cuda().to(dtype)
 tokens=a.embed_motion(z);velocity=a.output(tokens,1);action=a.output(tokens,0)
 assert tokens.shape==(1,n,512) and velocity.shape==z.shape and action.shape==(1,32,7)
 loss=(velocity.float()-z.float()).square().mean()+action.float().square().mean();loss.backward()
 assert a.motion_patch.weight.grad.norm()>0 and a.motion_unpatch.weight.grad.norm()>0 and a.action_head[0].weight.grad.norm()>0
 rows.append(dict(tokens=n,shared=list(tokens.shape),velocity=list(velocity.shape),action=list(action.shape),loss=float(loss),backward=True))
 del a,tokens,velocity,action,loss
report['adapters']=rows
refkeys={k:tuple(v.shape) for k,v in c.state_dict().items()};del c,z2,decoded;gc.collect();torch.cuda.empty_cache()
with patch('safetensors.torch.load_file',side_effect=AssertionError('reconstruction must not load official weights')):
 c=ReconstructionCodec('weights/MotionCrafter/geometry_motion_vae',frame_batch=2,cameras=['image']).cuda().to(dtype)
assert {k:tuple(v.shape) for k,v in c.state_dict().items()}==refkeys
z=z.detach().requires_grad_(True);decoded=c.decode(z);loss=(decoded.float()-head.float()).abs().mean();loss.backward()
assert decoded.shape==head.shape and torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.norm()>0
assert c.decoder.conv_out.weight.grad.norm()>0 and c.decoder_2.conv_out.weight.grad.norm()>0
assert all(p.grad is None for name in c.encoder_names for p in getattr(c,name).parameters())
report['reconstruction']=dict(decoded=list(decoded.shape),loss=float(loss),latent_grad=float(z.grad.norm()),decoder_grad=True,geometry_motion_codec_architecture_unchanged=True,official_weights_not_loaded=True)
report['status']='passed';(out/'codec_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
