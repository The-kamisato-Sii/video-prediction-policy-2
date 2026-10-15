"""Time the original full-size FM training path with synthetic RGB inputs.

Released joint-100k weights are a timing fixture, not baseline initialization.
Formal baseline reproduction must use history-10k + freshly initialized Action.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from fullsize_meanflow import SyntheticRGB
from vpp2.meanflow.runtime import load_fm
from vpp2.runtime import create_vpp2_wan21_14b
from vpp2.trainer import JointTrainer
from vpp2.utils.training_profiler import TrainingProfiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if (
        min(args.batch_size, args.accumulation) < 1
        or args.warmup_steps < 0
        or args.steps <= args.warmup_steps
    ):
        parser.error("Positive batch/accumulation and steps > warmup-steps are required")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(42)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    out = Path(args.output)
    if (out / "profiling/summary.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load("configs/robodojo/train_100k.yaml")
    cfg.output_dir = str(out)
    cfg.batch_size = args.batch_size
    cfg.gradient_accumulation_steps = args.accumulation
    cfg.num_workers = 0
    cfg.max_steps = args.steps
    cfg.warmup_steps = 0
    cfg.lr_tail = None
    cfg.log_every = 1
    cfg.save_every = args.steps + 1
    cfg.save_final_checkpoint = False
    cfg.eval_every = 0
    cfg.eval_at_start = False
    cfg.preflight.enabled = False
    cfg.wandb.enabled = False
    cfg.paths.wan = "weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P"
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    cfg.model.dit_checkpoint_path = None
    cfg.profiling = dict(
        enabled=True,
        skip_steps=args.warmup_steps,
        collect_steps=args.steps - args.warmup_steps,
        synchronize_cuda=True,
        log_every=1,
    )
    profiler = TrainingProfiler.from_config(cfg)
    kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    kwargs.pop("_target_")
    with profiler.measure_startup("model_load_ms", cuda=True):
        model = create_vpp2_wan21_14b(
            **kwargs, model_dtype=torch.bfloat16, device=f"cuda:{torch.cuda.current_device()}"
        )
        # Strict paired-weight loading only: no MeanFlow configuration or branches.
        provenance = load_fm(
            model,
            dict(
                format="deployment_pair",
                action_checkpoint="weights/robodojo_release/checkpoints/joint2b_s100000/action.pt",
                video_checkpoint="weights/robodojo_release/checkpoints/joint2b_s100000/video.pt",
            ),
            "robodojo",
        )
    assert not hasattr(model, "meanflow_config")
    global_batch = world * args.batch_size * args.accumulation
    dataset = SyntheticRGB(max(global_batch * args.steps * 2, 64), seed=42 + rank)
    trainer = JointTrainer(model, dataset, cfg=cfg, profiler=profiler)
    assert trainer.accelerator.num_processes == world
    assert trainer._deepspeed_zero_stage() == 2
    if rank == 0:
        (out / "scope.json").write_text(json.dumps(dict(
            scope="Synthetic throughput measurement of original FM; not baseline reproduction",
            objective="original_joint_flow_matching", world_size=world,
            micro_batch=args.batch_size, accumulation=args.accumulation,
            diagnostic_global_batch=global_batch, reference_global_batch=288,
            reference_updates=100000, zero_stage=trainer._deepspeed_zero_stage(),
            initialization=provenance,
            excluded_costs=["dataset decoding", "periodic evaluation", "checkpoint writes",
                            "data preparation", "interruptions"],
        ), indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    assert trainer.global_step == args.steps
    (out / f"memory_rank{rank}.json").write_text(json.dumps(dict(
        rank=rank, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
    ), indent=2) + "\n")
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        print("PASS: original FM throughput profile complete", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
