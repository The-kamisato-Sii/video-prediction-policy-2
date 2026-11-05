"""Check flatten/masking equivalence and stop-gradient JVP on both label shapes."""
import json
from pathlib import Path
import torch
from vpp2.meanflow.objectives import data_loss,improved_data_loss,directional_derivative

torch.manual_seed(42)
reports=[]
for shape in [(3,32,7),(3,8,8,28,28)]:
 clean=torch.randn(shape);noise=torch.randn_like(clean)
 valid=torch.rand(shape)>.2;valid[2]=False
 r=torch.tensor([.2,.6,.1]);t=torch.tensor([.8,.6,.9])
 x=clean.reshape(3,-1,1);n=noise.reshape_as(x);pad=~valid.reshape(3,-1)
 for kind in ['mf','boundary','auxiliary']:
  w=torch.nn.Parameter(torch.tensor(.3))
  def fn(z,lo,hi):return w*z+lo[:,None,None]+2*hi[:,None,None]
  def der(z,lo,hi,v,rows):
   return directional_derivative(fn,z[rows],lo[rows],hi[rows],v[rows])
  def pv(z,hi,rows):return fn(z[rows],hi[rows],hi[rows])
  opts=dict(baseline_reduction=True,loss_weighting='adaptive',distributed=False)
  if kind=='mf':loss,metrics=data_loss(fn,der,x,n,r,t,pad,**opts)
  else:loss,metrics=improved_data_loss(fn,der,pv,x,n,r,t,pad,auxiliary=kind=='auxiliary',**opts)
  # Independent analytic loss in native action/latent shape.
  e=lambda a:a.reshape(3,*([1]*(len(shape)-1)))
  z=clean+e(t)*(noise-clean);pred=w*z+e(r)+2*e(t)
  direction=noise-clean if kind=='mf' else (w.detach()*z+3*e(t)).detach()
  correction=e(t-r)*(w.detach()*direction+2)
  target=(noise-clean-correction).detach()
  def weighted(pred,target):
   per=torch.where(valid,(pred-target).square(),0).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
   return (per*(per.detach()+.001).pow(-.5)).mean()
  expected=weighted(pred,target)
  if kind=='auxiliary':expected=expected+weighted(w*z+3*e(t),noise-clean)
  torch.testing.assert_close(loss,expected)
  grad=torch.autograd.grad(loss,w,retain_graph=True)[0];gref=torch.autograd.grad(expected,w)[0]
  torch.testing.assert_close(grad,gref)
  assert metrics['mf/valid_samples']==2
  reports.append(dict(shape=list(shape),kind=kind,loss=float(loss.detach()),gradient=float(grad)))
# Original action-shaped objectives must be invariant to element-wise flattening.
clean=torch.randn(3,32,7);noise=torch.randn_like(clean);pad=torch.zeros(3,32,dtype=torch.bool);pad[0,24:]=True
for flatten in [False,True]:
 c=clean.reshape(3,-1,1) if flatten else clean;n=noise.reshape_as(c)
 p=pad[:,:,None].expand_as(clean).reshape(3,-1) if flatten else pad
 w=torch.nn.Parameter(torch.tensor(.3))
 fn=lambda z,lo,hi:w*z+lo[:,None,None]+2*hi[:,None,None]
 der=lambda z,lo,hi,v,rows:directional_derivative(fn,z[rows],lo[rows],hi[rows],v[rows])
 loss,_=data_loss(fn,der,c,n,r,t,p,baseline_reduction=True,loss_weighting='adaptive',distributed=False)
 g=torch.autograd.grad(loss,w)[0]
 if not flatten:ref=(loss.detach(),g)
 else:torch.testing.assert_close(loss,ref[0]);torch.testing.assert_close(g,ref[1])
out=Path('runs/libero_flow_meanflow_validation_20261104');out.mkdir(exist_ok=True)
(out/'math.json').write_text(json.dumps(dict(status='PASS',cases=reports,action_flatten_equivalence=True),indent=2)+'\n')
print('ANALYTIC_MF_IMF_AND_MASK_GRADIENTS_PASS')
