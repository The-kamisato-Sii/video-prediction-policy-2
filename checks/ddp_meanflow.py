"""Two CPU ranks verify wrapped forward synchronization, including learned split parameters."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from vpp2.meanflow.objectives import SplitGenerator, data_loss


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.generator = SplitGenerator()

    def forward(self, clean, noise, r, t, pad):
        def derivative(z, r, t, v, rows):
            return (v[rows] * (1 + t[rows, None, None])).detach()

        return data_loss(
            lambda z, r, t: self.linear(z),
            derivative,
            clean,
            noise,
            r,
            t,
            pad,
            objective="kmf",
            generator=self.generator,
        )[0]


if __name__ == "__main__":
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(10)
    model = DDP(Policy())
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    torch.manual_seed(42 + rank)
    for step in range(6):
        x = torch.randn(3, 4, 2)
        noise = torch.randn_like(x)
        pad = torch.zeros(3, 4, dtype=torch.bool)
        if rank == 1:
            pad[1:] = True
        t = torch.full((3,), 0.5)
        r = torch.zeros(3)
        # Alternate mixed-rank and globally diagonal intervals, then all-padding.
        if step in (1, 4) and rank == 0 or step == 2:
            r = t.clone()
        if step == 3:
            pad[:] = True
        loss = model(x, noise, r, t, pad)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        copies = [torch.zeros_like(p) for _ in range(dist.get_world_size())]
        dist.all_gather(copies, p.detach())
        for other in copies:
            torch.testing.assert_close(other, p, rtol=0, atol=0)
    if rank == 0:
        print("DDP policy and split-generator synchronization passed", flush=True)
    dist.destroy_process_group()
