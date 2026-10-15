"""Exercise actual trainers, parameter groups, accumulation and Accelerate state resume."""

import argparse
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf
from test_meanflow import tiny_model

from vpp2.libero.trainer import LiberoTrainer
from vpp2.meanflow.model import MeanFlowLibero, MeanFlowVPP2
from vpp2.trainer import JointTrainer


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sample = dict(
            context=torch.randn(3, 16),
            context_mask=torch.ones(3, dtype=torch.bool),
            proprio=torch.randn(3),
            action=torch.randn(4, 3),
            input_latents=torch.randn(16, 2, 2, 2),
            condition_latents=torch.randn(16, 1, 2, 2),
            action_is_pad=torch.zeros(4, dtype=torch.bool),
            action_loss_is_pad=torch.zeros(4, dtype=torch.bool),
        )

    def __len__(self):
        return 16

    def __getitem__(self, i):
        return self.sample


def build(self, sample, **kwargs):
    sample = {k: v.to(self.device) for k, v in sample.items()}
    c, m = self._append_proprio_to_context(
        sample["context"], sample["context_mask"], sample["proprio"]
    )
    return dict(
        sample,
        context=c,
        context_mask=m,
        clip_fea=None,
        fuse_vae_embedding_in_latents=False,
        first_frame_latents=None,
        image_is_pad=None,
        num_video_frames=5,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["robodojo", "libero"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--diagonal", type=float, default=0.5)
    args = p.parse_args()
    robo = args.benchmark == "robodojo"
    cfg = OmegaConf.load(
        "configs/robodojo/train_100k.yaml" if robo else "configs/libero/train_action.yaml"
    )
    cfg.output_dir = args.output
    cfg.batch_size = 2
    cfg.gradient_accumulation_steps = 2
    cfg.num_workers = 0
    cfg.max_steps = args.steps
    cfg.eval_every = 0
    cfg.eval_at_start = False
    cfg.save_every = 2
    cfg.log_every = 1
    cfg.warmup_steps = 0
    cfg.learning_rate = 0.001
    cfg.mixed_precision = "no"
    cfg.resume = args.resume
    cfg.preflight.enabled = False
    cfg.wandb.enabled = False
    if robo:
        cfg.joint_learning_rates.video = 0.001
        cfg.joint_learning_rates.action = 0.001
    model = tiny_model(
        MeanFlowVPP2 if robo else MeanFlowLibero,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.build_inputs = types.MethodType(build, model)
    model.configure_meanflow(
        dict(
            objective="kmf",
            split="learned",
            derivative_backend="jvp",
            budget_steps=4,
            interval=dict(diagonal_probability=args.diagonal),
        ),
        {"synthetic": True},
    )
    if not robo:
        video = Path(args.output).parent / "fake_video.pt"
        if not video.exists():
            torch.save({"dit": model.video_expert.state_dict(), "step": 10000}, video)
        cfg.frozen_video_checkpoint = str(video)
    cls = JointTrainer if robo else LiberoTrainer
    trainer = cls(model, Dataset(), cfg=cfg)
    groups = (
        trainer.model.optimizer.bit16_groups
        if trainer._deepspeed_zero_stage()
        else [g["params"] for g in trainer.optimizer.param_groups]
    )
    grouped = [p for group in groups for p in group]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters() if p.requires_grad}
    generator_before = {
        k: v.detach().clone() for k, v in model.split_generator.state_dict().items()
    }
    trainer.train()
    if args.diagonal == 1.0:
        for k, v in model.split_generator.state_dict().items():
            torch.testing.assert_close(v, generator_before[k], rtol=0, atol=0)
    assert trainer.global_step == args.steps
    print(args.benchmark, "trainer groups, accumulation and state saved PASS", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
