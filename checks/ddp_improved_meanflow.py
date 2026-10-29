"""Two ranks: unequal diagonal/padding cases, fixed collective order and rank0 eval."""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from vpp2.meanflow.objectives import improved_data_loss, directional_derivative

class Policy(nn.Module):
    def __init__(self, auxiliary):
        super().__init__();self.u=nn.Linear(2,2);self.auxiliary=auxiliary
        if auxiliary:self.v=nn.Linear(2,2)
    def forward(self,x,e,r,t,pad,distributed=True):
        predict=lambda z,r,t:self.u(z)+t[:,None,None]
        def derivative(z,r,t,v,rows):
            return directional_derivative(predict,z[rows],r[rows],t[rows],v[rows])
        return improved_data_loss(predict,derivative,
            lambda z,t,rows:(self.v if self.auxiliary else self.u)(z[rows]),x,e,r,t,pad,
            auxiliary=self.auxiliary,distributed=distributed)[0]

if __name__=='__main__':
    dist.init_process_group('gloo');rank=dist.get_rank()
    for auxiliary in [False,True]:
        torch.manual_seed(10);model=DDP(Policy(auxiliary));opt=torch.optim.AdamW(model.parameters(),lr=.001)
        torch.manual_seed(42+rank)
        for step in range(6):
            x=torch.randn(3,4,2);e=torch.randn_like(x);t=torch.full((3,),.6);r=torch.zeros_like(t)
            pad=torch.zeros(3,4,dtype=torch.bool)
            if step==0 and rank==0 or step==1:r=t.clone()
            if step==2 and rank==1 or step==3:pad[:]=True
            loss=model(x,e,r,t,pad);assert torch.isfinite(loss);loss.backward();opt.step();opt.zero_grad()
        for p in model.parameters():
            copies=[torch.zeros_like(p) for _ in range(2)];dist.all_gather(copies,p)
            torch.testing.assert_close(copies[0],copies[1],rtol=0,atol=0)
        if rank==0:
            with torch.no_grad():assert torch.isfinite(model.module(x,e,r,t,pad,distributed=False))
        dist.barrier()
    if rank==0:print('IMF_DDP_PASS: boundary/auxiliary, uneven diagonal/padding, rank0 eval',flush=True)
    dist.destroy_process_group()
