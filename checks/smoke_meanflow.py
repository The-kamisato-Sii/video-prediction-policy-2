"""Synthetic 40-layer engineering probe. Does not measure full-size model performance."""

import argparse
import json
import time
import types
from pathlib import Path

import torch
from test_meanflow import tiny_model

from vpp2.libero.action_model import LiberoActionModel
from vpp2.meanflow.model import MeanFlowLibero, MeanFlowVPP2
from vpp2.models.wan21_14b.vpp2 import VPP2


def sample_inputs(self, sample, **kwargs):
    c, mask = self._append_proprio_to_context(
        sample["context"], sample["context_mask"], sample["proprio"]
    )
    return dict(sample, context=c, context_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", default="runs/meanflow_engineering_probe")
    p.add_argument("--layers", type=int, default=40)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for benchmark, cls, teacher_cls in [
        ("robodojo", MeanFlowVPP2, VPP2),
        ("libero", MeanFlowLibero, LiberoActionModel),
    ]:
        for objective, split in [
            ("teacher_trajectory", "midpoint"),
            ("data_meanflow", "midpoint"),
            ("kmf", "midpoint"),
            ("kmf", "learned"),
        ]:
            torch.manual_seed(42)
            model = tiny_model(cls, args.layers, args.device, torch.bfloat16, checkpoint=True)
            model.build_inputs = types.MethodType(sample_inputs, model)
            teacher = None
            if objective == "teacher_trajectory":
                teacher = tiny_model(teacher_cls, args.layers, args.device, torch.bfloat16)
                teacher.load_state_dict(model.state_dict(), strict=True)
                teacher.build_inputs = types.MethodType(sample_inputs, teacher)
                teacher.eval().requires_grad_(False)
            cfg = dict(
                objective=objective,
                split=split,
                derivative_backend="jvp",
                budget_steps=args.steps,
                interval=dict(
                    full_interval_probability=0.5, min_gap=1e-4, diagonal_probability=0.5
                ),
                teacher=dict(steps=10, sigma_shift=1.0 if benchmark == "robodojo" else 5.0),
            )
            model.configure_meanflow(cfg, {"synthetic": True}, teacher)
            model.eval().requires_grad_(False)
            model.action_expert.train().requires_grad_(True)
            model.proprio_encoder.train().requires_grad_(True)
            if benchmark == "robodojo":
                model.video_expert.train().requires_grad_(True)
            if model.split_generator is not None:
                model.split_generator.requires_grad_(True)
            b = 2
            d = args.device
            dt = torch.bfloat16
            sample = dict(
                context=torch.randn(b, 3, 16, device=d, dtype=dt),
                context_mask=torch.ones(b, 3, device=d, dtype=torch.bool),
                proprio=torch.randn(b, 3, device=d, dtype=dt),
                action=torch.randn(b, 4, 3, device=d, dtype=dt),
                input_latents=torch.randn(b, 16, 2, 2, 2, device=d, dtype=dt),
                condition_latents=torch.randn(b, 16, 1, 2, 2, device=d, dtype=dt),
                clip_fea=None,
                fuse_vae_embedding_in_latents=False,
                action_is_pad=torch.zeros(b, 4, device=d, dtype=torch.bool),
                action_loss_is_pad=torch.tensor(
                    [[False, False, False, True], [False, False, True, True]], device=d
                ),
                image_is_pad=None,
                first_frame_latents=None,
                num_video_frames=5,
            )
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=1e-3)
            initial = model.action_expert.head.weight.detach().clone()
            losses = []
            timings = []
            if d == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(args.steps):
                model.meanflow_update_step.fill_(step)
                if d == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                loss, metrics = model(sample)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
                assert model.proprio_encoder.weight.grad is not None
                if teacher is not None:
                    assert all(p.grad is None for p in teacher.parameters())
                if benchmark == "libero":
                    assert all(p.grad is None for p in model.video_expert.parameters())
                else:
                    assert any(
                        p.grad is not None and p.grad.norm() > 0
                        for p in model.video_expert.parameters()
                    )
                opt.step()
                opt.zero_grad(set_to_none=True)
                if d == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                losses.append(float(loss))
            assert not torch.equal(initial, model.action_expert.head.weight)
            ckpt = out / f"{benchmark}_{objective}_{split}.pt"
            model.save_checkpoint(ckpt, opt, args.steps)
            saved = model.action_expert.head.weight.detach().clone()
            with torch.no_grad():
                model.action_expert.head.weight.zero_()
            assert model.load_checkpoint(ckpt, opt) == args.steps
            torch.testing.assert_close(model.action_expert.head.weight, saved, rtol=0, atol=0)
            loss, _ = model(sample)
            loss.backward()
            opt.step()
            row = dict(
                benchmark=benchmark,
                objective=objective,
                split=split,
                steps=args.steps,
                layers=args.layers,
                synthetic_reduced_width=True,
                losses=losses,
                step_seconds=timings,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if d == "cuda" else None,
                checkpoint_roundtrip=True,
                resumed_update=True,
                final_metrics=metrics,
            )
            results.append(row)
            (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(benchmark, objective, split, "PASS", losses[0], losses[-1], flush=True)
            del opt, model, teacher, params, loss
            if d == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
