"""Actual shared loss adapter vs unchunked objectives; includes >4 JVP rows."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from vpp2.libero_flow_meanflow.model import SharedMeanFlowMixin
from vpp2.meanflow.objectives import data_loss,improved_data_loss,directional_derivative,sample_intervals
class Probe(nn.Module):
 _meanflow_loss=SharedMeanFlowMixin._meanflow_loss
 def __init__(self,kind):
  super().__init__();self.w=nn.Parameter(torch.tensor(.3));self.v=nn.Parameter(torch.tensor(.2))
  self.action_expert=nn.Identity();self.action_expert.hidden_dim=512;self.action_expert.eval()
  self.flow_adapters=SimpleNamespace(tokens=32,pt=4,ratio=1)
  self.train_action_scheduler=SimpleNamespace(shift=5.)
  self.meanflow_config=dict(objective='data_meanflow' if kind=='mf' else 'data_improved_meanflow',
   interval=dict(diagonal_probability=0.,diagonal_time_sampling='independent'),loss_weighting='adaptive',
   adaptive_weighting=dict(power=.5,c=.001),derivative_backend='jvp',imf={'auxiliary_loss_weight':1.})
  self.auxiliary=kind=='aux';self.last_shapes={}
 @staticmethod
 def _slice_conditions(c,rows):return None
 def _mean_velocity(self,z,indicator,r,t,c,branch='u'):
  sh=(-1,)+ (1,)*(z.ndim-1)
  return (self.w if branch=='u' else self.v)*z+r.reshape(sh)+2*t.reshape(sh)
rows=[]
for indicator,shape in [(0,(9,32,7)),(1,(9,8,8,28,28))]:
 for kind in ['mf','boundary','aux']:
  m=Probe(kind);torch.manual_seed(12);clean=torch.randn(shape);valid=torch.rand(shape)>.15;valid[-1]=False
  torch.manual_seed(42);loss,_=m._meanflow_loss(clean,valid,indicator,None)
  g=torch.autograd.grad(loss, (m.w,m.v),allow_unused=True)
  torch.manual_seed(42);x=clean.reshape(9,-1,1);noise=torch.randn_like(x)
  t,r=sample_intervals(9,'cpu',5.,0.,'independent');pad=~valid.reshape(9,-1)
  fn=lambda z,a,b:m.w*z+a[:,None,None]+2*b[:,None,None]
  der=lambda z,a,b,v,sel:directional_derivative(fn,z[sel],a[sel],b[sel],v[sel])
  pv=lambda z,t,sel:(m.v if kind=='aux' else m.w)*z[sel]+3*t[sel,None,None]
  opt=dict(baseline_reduction=True,loss_weighting='adaptive',distributed=False)
  ref,_=data_loss(fn,der,x,noise,r,t,pad,**opt) if kind=='mf' else improved_data_loss(fn,der,pv,x,noise,r,t,pad,auxiliary=kind=='aux',**opt)
  gr=torch.autograd.grad(ref,(m.w,m.v),allow_unused=True)
  torch.testing.assert_close(loss,ref)
  for a,b in zip(g,gr):
   if a is None:assert b is None
   else:torch.testing.assert_close(a,b)
  rows.append(dict(indicator=indicator,kind=kind,batch=9,status='PASS'))
p=Path('runs/libero_flow_meanflow_validation_20261104/adapter.json');p.write_text(json.dumps(dict(status='PASS',cases=rows),indent=2)+'\n')
print('CHUNKED_ADAPTER_LOSS_AND_GRADIENTS_PASS')
