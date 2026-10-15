"""Full-size RoboDojo training smoke with released weights and synthetic RGB.

Uses the real input encoders, objective, trainer and optimizer. This is neither a
real-dataset run nor a convergence/benchmark test. Launch through Accelerate with
ZeRO-2 on eight sufficiently large GPUs; no large checkpoint is written.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vpp2.meanflow.cli import load_config
from vpp2.trainer import JointTrainer


class SyntheticRGB(torch.utils.data.Dataset):
    def __init__(self, length, seed):
        self.length = length
        rng = torch.Generator().manual_seed(seed)
        # The recipe samples [0,4,...,64]: 65 native steps become 17 RGB frames.
        video = torch.rand(3, 17, 240, 416, generator=rng) * 2 - 1
        self.sample = dict(
            video=video,
            condition_video=video[:, :9],
            context=torch.randn(128, 4096, generator=rng),
            context_mask=torch.ones(128, dtype=torch.bool),
            proprio=torch.randn(1, 16, generator=rng),
            action=torch.randn(32, 16, generator=rng),
            action_is_pad=torch.zeros(32, dtype=torch.bool),
            image_is_pad=torch.zeros(17, dtype=torch.bool),
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["mf", "kmf_learned", "kmf_midpoint", "distill"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"rank{rank}.json").exists():
        raise FileExistsError("Choose a fresh --output directory")
    cfg = load_config(f"configs/meanflow/robodojo_{args.method}.yaml")
    cfg.output_dir = str(out)
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.num_workers = 0
    cfg.max_steps = args.steps
    cfg.warmup_steps = 0
    cfg.lr_tail = None
    cfg.save_every = args.steps + 1
    cfg.save_final_checkpoint = False
    cfg.eval_every = 0
    cfg.eval_at_start = False
    cfg.log_every = 1
    cfg.preflight.enabled = False
    cfg.wandb.enabled = False
    # Exercise off-diagonal derivatives at every update in this diagnostic.
    cfg.meanflow.interval.diagonal_probability = 0.0
    cfg.meanflow.budget_steps = args.steps
    if rank == 0:
        OmegaConf.save(cfg, out / "diagnostic_config.yaml")
    start = time.monotonic()
    print(f"rank={rank} loading original 40-layer Video/Action weights", flush=True)
    model = instantiate(
        cfg.model, model_dtype=torch.bfloat16, device=f"cuda:{torch.cuda.current_device()}"
    )
    records = []

    def forward_record(module, inputs, output):
        loss, metrics = output
        values = dict(loss=float(loss.detach()), **metrics)
        assert all(math.isfinite(v) for v in values.values()), values
        records.append(values)

    model.register_forward_hook(forward_record)
    trainer = JointTrainer(
        model, SyntheticRGB(max(world * args.steps * 2, 32), 42 + rank), cfg=cfg
    )
    assert trainer.accelerator.num_processes == world
    assert trainer._deepspeed_zero_stage() == 2, "This diagnostic requires ZeRO-2"
    assert all(p.requires_grad for p in model.video_expert.parameters())
    assert all(p.requires_grad for p in model.action_expert.parameters())
    gradients = {}
    handles = []
    for name, param in [
        ("video", model.video_expert.patch_embedding.weight),
        ("action", model.action_expert.head.weight),
        ("proprio", model.proprio_encoder.weight),
    ]:
        def record_gradient(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients.setdefault(name, []).append(float(grad.float().norm()))
        handles.append(param.register_hook(record_gradient))
    initial = model.action_expert.head.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.monotonic()
    print(f"rank={rank} starting real trainer with synthetic RGB", flush=True)
    trainer.train()
    torch.cuda.synchronize()
    assert trainer.global_step == args.steps
    assert len(records) == args.steps
    assert all(
        name in gradients and any(v > 0 for v in gradients[name])
        for name in ("video", "action", "proprio")
    ), gradients
    delta = (
        (model.action_expert.head.weight.detach().float() - initial.float()).abs().max().item()
    )
    assert delta > 0, "Action weights did not update"
    if model._teacher is not None:
        assert all(p.grad is None and not p.requires_grad for p in model._teacher.parameters())
    report = dict(
        status="PASS",
        scope="full-size synthetic RGB training; no dataset or convergence validation",
        method=args.method,
        world_size=world,
        rank=rank,
        zero_stage=2,
        micro_batch=1,
        accumulation=1,
        global_batch=world,
        updates=args.steps,
        video_shape=[1, 3, 17, 240, 416],
        action_shape=[1, 32, 16],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        load_and_setup_seconds=train_start - start,
        train_seconds=time.monotonic() - train_start,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        action_head_max_change=delta,
        gradient_norms=gradients,
        records=records,
        provenance=model.meanflow_provenance,
    )
    (out / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"rank={rank} PASS: {args.steps} full-size updates", flush=True)
    trainer.accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
