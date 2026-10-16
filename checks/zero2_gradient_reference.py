"""Check ZeRO-1/2 gradient averaging/accumulation against plain PyTorch.

Run with torchrun --nproc_per_node=8. Covers shared parameters, non-reentrant
activation checkpointing, a frozen branch, and accumulation=1/3 in FP32/BF16.
This checks distributed arithmetic on a small model, not full VPP2 convergence.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import deepspeed
from deepspeed.utils import safe_get_full_grad


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(16, 16).requires_grad_(False)
        self.shared = torch.nn.Linear(16, 32)
        self.output = torch.nn.Linear(32, 8)

    def forward(self, x):
        with torch.no_grad():
            context = self.frozen(x)
        first = checkpoint(self.shared, context, use_reentrant=False)
        second = checkpoint(self.shared, context * 0.5, use_reentrant=False)
        return self.output(torch.nn.functional.silu(first + second))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], required=True)
    parser.add_argument('--accumulation', type=int, required=True)
    parser.add_argument('--zero-stage', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed()
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    torch.manual_seed(79)
    model = Network().cuda().to(dtype=dtype)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    config = dict(train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=args.accumulation,
                  train_batch_size=2*world*args.accumulation, gradient_clipping=0.0,
                  bf16=dict(enabled=args.precision == 'bf16'),
                  zero_optimization=dict(stage=args.zero_stage, contiguous_gradients=False, overlap_comm=False,
                                         reduce_bucket_size=200000000, allgather_bucket_size=200000000))
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    gradient_hook_count = len(engine.optimizer._grad_acc_hooks)
    if args.zero_stage == 1:
        assert gradient_hook_count == 0, gradient_hook_count
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    expected = {n: torch.zeros_like(p, dtype=torch.float32) for n,p in reference.named_parameters() if p.requires_grad}
    for micro in range(args.accumulation):
        generator = torch.Generator(device='cuda').manual_seed(1000 + rank*100 + micro)
        x = torch.randn(2,16,device='cuda',dtype=dtype,generator=generator)
        y = torch.randn(2,8,device='cuda',dtype=dtype,generator=generator)
        reference.zero_grad(set_to_none=True)
        ref_loss = (reference(x).float() - y.float()).square().mean() / args.accumulation
        ref_loss.backward()
        for name, param in reference.named_parameters():
            if param.requires_grad:
                expected[name].add_(param.grad.float())
        loss = (engine(x).float() - y.float()).square().mean()
        engine.set_gradient_accumulation_boundary(micro == args.accumulation-1)
        engine.backward(loss)
    comparisons = {}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None
            continue
        ref = expected[name]
        dist.all_reduce(ref)
        ref.div_(world)
        actual = safe_get_full_grad(param).float()
        error = (actual-ref).norm()/ref.norm().clamp_min(1e-12)
        comparisons[name] = dict(relative_l2=float(error), max_absolute=float((actual-ref).abs().max()))
        assert error < (0.025 if dtype == torch.bfloat16 else 2e-5), comparisons
    engine.step()
    assert any(not torch.equal(param, before[name]) for name, param in model.named_parameters() if param.requires_grad)
    assert all(torch.equal(param, before[name]) for name, param in model.named_parameters() if not param.requires_grad)
    hashes = [torch.empty_like(model.output.weight) for _ in range(world)]
    dist.all_gather(hashes, model.output.weight)
    assert all(torch.equal(hashes[0], value) for value in hashes)
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(pass_check=True, world_size=world,
            deepspeed=deepspeed.__version__, torch=torch.__version__, precision=args.precision,
            zero_stage=args.zero_stage, gradient_hook_count=gradient_hook_count,
            gradient_accumulation_dtype=str(engine.optimizer.gradient_accumulation_dtype),
            accumulation=args.accumulation, gradients=comparisons, updated_replicas_identical=True),indent=2)+'\n')
        print('PASS gradient reference', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
