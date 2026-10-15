"""Shared objectives with benchmark-specific condition and gradient boundaries."""

from contextlib import contextmanager, nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vpp2.libero.action_model import LiberoActionModel
from vpp2.models.wan21_14b.vpp2 import VPP2

from .objectives import (
    SplitGenerator,
    data_loss,
    directional_derivative,
    masked_mse,
    sample_intervals,
    teacher_intervals,
    valid_elements,
)
from .teacher import rollout


def prepare_condition(model, inputs, noise_video, *, joint=False):
    """Pure-noise Video prefill; joint keeps its graph and returns the Video prediction."""
    time = torch.full(
        (noise_video.shape[0],),
        model.train_video_scheduler.num_train_timesteps,
        device=noise_video.device,
        dtype=noise_video.dtype,
    )
    with nullcontext() if joint else torch.no_grad():
        pre = model.video_expert.pre_dit(
            x=noise_video,
            timestep=time,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            clip_fea=inputs["clip_fea"],
            condition_latents=inputs["condition_latents"],
        )
        n = pre["tokens"].shape[1]
        mask = model._build_mot_attention_mask(
            n, inputs["action"].shape[1], pre["meta"]["tokens_per_frame"], noise_video.device
        )
        if mask[:n, n:].any():
            raise ValueError("Video must not attend to action tokens")
        with nullcontext() if joint else model._disable_video_cache_checkpointing():
            kv, final = model.mot.prefill_video_cache(
                pre["tokens"],
                pre["freqs"],
                pre["t_mod"],
                {"context": pre["context"], "mask": pre["context_mask"]},
                mask[:n, :n],
                return_final_tokens=True,
            )
        prediction = model.video_expert.post_dit(final, pre) if joint else None
    return dict(
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        video_kv_cache=kv,
        attention_mask=mask,
        video_seq_len=n,
    ), prediction


def slice_condition(condition, rows):
    return dict(
        condition,
        context=condition["context"].detach()[rows],
        context_mask=condition["context_mask"][rows],
        video_kv_cache=[
            {k: v.detach()[rows] for k, v in layer.items()} for layer in condition["video_kv_cache"]
        ],
    )


@contextmanager
def derivative_mode(model):
    # No activation-checkpoint autograd Functions in forward AD. Preserve primary settings.
    saved = []
    for module in model.modules():
        for key in ("use_gradient_checkpointing", "mot_checkpoint_mixed_attn"):
            if hasattr(module, key):
                saved.append((module, key, getattr(module, key)))
                setattr(module, key, False)
    try:
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            yield
    finally:
        for module, key, value in saved:
            setattr(module, key, value)


class MeanFlowMixin:
    def configure_meanflow(self, config, provenance, teacher=None):
        self.meanflow_config = dict(config)
        self.meanflow_provenance = provenance
        self.action_sampler = "meanflow"
        self.action_expert.enable_meanflow()
        self.split_generator = None
        if config["objective"] == "kmf" and config.get("split", "midpoint") == "learned":
            self.split_generator = SplitGenerator().to(device=self.device, dtype=torch.float32)
        # Frozen teacher is deliberately outside student state_dict / optimizer / DDP.
        # It is constructed independently on each rank, with fixed weights and its own proprio.
        object.__setattr__(self, "_teacher", teacher)
        self.register_buffer(
            "meanflow_update_step", torch.zeros((), dtype=torch.long), persistent=True
        )

    def predict_action_velocity(self, z, r, t, condition):
        return self.predict_action_velocity_with_cache(
            latents_action=z, timestep_action=t * 1000.0, timestep_end=r * 1000.0, **condition
        )

    def training_loss(self, sample, tiled=False, training_stage=None, **kwargs):
        joint = self.meanflow_benchmark == "robodojo"
        inputs = self.build_inputs(
            sample, tiled=tiled, training_stage="joint" if joint else "action"
        )
        action = inputs["action"]
        pad = inputs.get("action_loss_is_pad", inputs["action_is_pad"])
        latent = inputs["input_latents"]
        if latent is not None:
            noise_video = torch.randn_like(latent)
        else:
            cond = inputs["condition_latents"]
            frames = (inputs["num_video_frames"] - 1) // int(
                self.vae.temporal_downsample_factor
            ) + 1
            noise_video = torch.randn(
                cond.shape[0],
                cond.shape[1],
                frames,
                *cond.shape[-2:],
                device=cond.device,
                dtype=cond.dtype,
            )
        noise = torch.randn_like(action)
        condition, video_prediction = prepare_condition(self, inputs, noise_video, joint=joint)
        cfg = self.meanflow_config
        kind = cfg["objective"]
        interval = cfg["interval"]
        metrics = {}
        if kind == "teacher_trajectory":
            teacher = self._teacher
            if teacher is None:
                raise RuntimeError("Training requires the independent frozen teacher")
            with torch.no_grad():
                teacher.eval()
                teacher_inputs = teacher.build_inputs(
                    sample, tiled=tiled, training_stage="joint" if joint else "action"
                )
                teacher_condition, _ = prepare_condition(teacher, teacher_inputs, noise_video)
                states, sigmas = rollout(
                    teacher,
                    noise,
                    teacher_condition,
                    cfg["teacher"]["steps"],
                    cfg["teacher"]["sigma_shift"],
                )
                z, r, t, target = teacher_intervals(
                    states, sigmas, interval["full_interval_probability"], interval["min_gap"]
                )
                del teacher_condition
            pred = self.predict_action_velocity(z, r, t, condition)
            action_loss = masked_mse(pred, target, pad)
            endpoint_rows = (t == 1) & (r == 0)
            metrics["teacher_endpoint_mse"] = (
                masked_mse(
                    (z.float() - pred.float())[endpoint_rows],
                    states[-1].float()[endpoint_rows],
                    None if pad is None else pad[endpoint_rows],
                    distributed=False,
                )
                if endpoint_rows.any()
                else action_loss.detach() * 0
            )
            metrics["teacher_nfe"] = cfg["teacher"]["steps"]
        else:
            t, r = sample_intervals(
                action.shape[0],
                action.device,
                self.train_action_scheduler.shift,
                interval["diagonal_probability"],
            )

            def predict(z, r, t):
                return self.predict_action_velocity(z, r, t, condition)

            def derivative(z, r, t, v, rows):
                fixed = slice_condition(condition, rows)
                with derivative_mode(self):
                    return directional_derivative(
                        lambda x, lo, hi: self.predict_action_velocity(x, lo, hi, fixed),
                        z[rows],
                        r[rows],
                        t[rows],
                        v[rows],
                        cfg["derivative_backend"],
                        cfg.get("finite_difference_step", 0.01),
                    )

            progress = min(1.0, float(self.meanflow_update_step) / max(1, cfg["budget_steps"]))
            action_loss, metrics = data_loss(
                predict,
                derivative,
                action,
                noise,
                r,
                t,
                pad,
                objective=kind,
                generator=self.split_generator,
                gamma=-0.5 + progress,
                alpha=cfg.get("generator_alpha", 0.1),
            )
            metrics["teacher_nfe"] = 0
        video_loss = action_loss.new_zeros(())
        if joint:
            if latent is None or self.train_video_scheduler.train_sampling_strategy != "pure_noise":
                raise ValueError(
                    "RoboDojo MeanFlow requires the original pure-noise Video objective"
                )
            target_video = noise_video - latent
            per_sample = self._compute_video_loss_per_sample(
                video_prediction,
                target_video,
                inputs["image_is_pad"],
                include_initial_video_step=inputs["first_frame_latents"] is None,
            )
            video_loss = per_sample.mean()
        total = self.loss_lambda_action * action_loss + self.loss_lambda_video * video_loss
        metrics.update(
            loss_action=action_loss.detach(),
            loss_video=video_loss.detach(),
            valid_action_elements=valid_elements(action, pad).sum(),
            empty_samples=(~valid_elements(action, pad).flatten(1).any(1)).sum(),
            generator_active=((t > r) & valid_elements(action, pad).flatten(1).any(1)).any(),
            time_mean=t.mean(),
            gap_mean=(t - r).mean(),
            diagonal_fraction=(t == r).float().mean(),
            full_interval_fraction=((t == 1) & (r == 0)).float().mean(),
            student_nfe=1,
        )
        return total, {
            k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()
        }

    def meanflow_metadata(self, step):
        return dict(
            version=1,
            objective=self.meanflow_config["objective"],
            action_sampler="meanflow",
            action_nfe=1,
            student_update_step=step,
            provenance=self.meanflow_provenance,
            config=self.meanflow_config,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(
            mot=self.mot.state_dict(),
            proprio_encoder=self.proprio_encoder.state_dict(),
            step=step,
            meanflow=self.meanflow_metadata(step),
            meanflow_update_step=torch.tensor(
                step if step is not None else int(self.meanflow_update_step)
            ),
            split_generator=None
            if self.split_generator is None
            else self.split_generator.state_dict(),
        )
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if payload.get("meanflow", {}).get("objective") != self.meanflow_config["objective"]:
            raise ValueError(
                "Resume objective differs; use explicit FM initialization for new experiments"
            )
        self.mot.load_state_dict(payload["mot"], strict=True)
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if self.split_generator is not None:
            self.split_generator.load_state_dict(payload["split_generator"], strict=True)
        self.meanflow_update_step.copy_(
            payload.get("meanflow_update_step", torch.tensor(payload["step"]))
        )
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload["step"]

    def save_action_checkpoint(
        self, path, video_checkpoint_path, step=None, resolved_cfg=None, **kwargs
    ):
        # LIBERO full resumable state is saved by Accelerate; export omits the train-only generator.
        payload = super().save_action_checkpoint(
            path, video_checkpoint_path, step, resolved_cfg, **kwargs
        )
        payload["meanflow"] = self.meanflow_metadata(step)
        torch.save(payload, path)
        return payload


class MeanFlowVPP2(MeanFlowMixin, VPP2):
    meanflow_benchmark = "robodojo"


class MeanFlowLibero(MeanFlowMixin, LiberoActionModel):
    meanflow_benchmark = "libero"
