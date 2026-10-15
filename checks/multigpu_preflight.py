"""Run with torchrun to verify NCCL collectives and BF16 gradients on every GPU."""

import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == args.expected_world_size, (world, args.expected_world_size)
    total = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(total)
    assert total.item() == world * (world + 1) / 2
    torch.manual_seed(42 + rank)
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.randn_like(x)
    loss = (x @ y).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    dist.all_reduce(x.grad)
    assert torch.isfinite(x.grad).all()
    row = dict(
        rank=rank,
        local_rank=local_rank,
        hostname=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),
        memory_gib=torch.cuda.get_device_properties(local_rank).total_memory / 2**30,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=list(torch.cuda.nccl.version()),
        all_reduce_sum=total.item(),
        bf16_loss=loss.item(),
        finite_nonzero_gradient=True,
    )
    rows = [None] * world
    dist.all_gather_object(rows, row)
    if rank == 0:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(dict(status="PASS", world_size=world, ranks=rows), indent=2) + "\n"
        )
        print(f"PASS: {world} GPUs, NCCL sum/gradient collectives, BF16 backward", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
