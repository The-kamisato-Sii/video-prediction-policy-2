"""Shared pretrained ActionDiT, task indicator, and isolated task adapters."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from .views import check_checkpoint_views

from vpp2.libero.action_model import LiberoActionModel
from vpp2.models.wan21_14b.wan_video_dit import sinusoidal_embedding_1d


class FlowActionAdapters(nn.Module):
    def __init__(self, dim=512, tokens=32, action_head_dim=128, motion_channels=8):
        super().__init__()
        if tokens not in (32, 64, 128):
            raise ValueError("tokens must be32,64,128")
        self.tokens, self.ratio, self.pt = tokens, tokens // 32, 128 // tokens
        if motion_channels not in (8, 16):
            raise ValueError("motion_channels must be 8 or 16")
        self.motion_channels = motion_channels
        self.indicator = nn.Embedding(2, dim)
        self.position = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.motion_patch = nn.Conv3d(motion_channels, dim, (self.pt, 7, 7), (self.pt, 7, 7), padding=0)
        self.motion_unpatch = nn.ConvTranspose3d(dim, motion_channels, (self.pt, 7, 7), (self.pt, 7, 7), padding=0)
        self.action_head = nn.Sequential(
            nn.Conv1d(dim, action_head_dim, self.ratio, stride=self.ratio, padding=0),
            nn.SiLU(), nn.Conv1d(action_head_dim, 7, 1))
        nn.init.normal_(self.indicator.weight, std=0.02)
        nn.init.normal_(self.motion_unpatch.weight, std=0.001)
        nn.init.zeros_(self.motion_unpatch.bias)

    def embed_motion(self, state):
        x = self.motion_patch(state)
        if tuple(x.shape[2:]) != (8 // self.pt, 4, 4):
            raise ValueError(f"Unexpected patch grid{tuple(x.shape)}")
        return x.flatten(2).transpose(1, 2)

    def output(self, hidden, indicator):
        if hidden.shape[1] != self.tokens:
            raise ValueError("Wrong shared sequence length")
        if indicator == 0:
            return self.action_head(hidden.transpose(1, 2)).transpose(1, 2)
        if indicator == 1:
            x = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], 8 // self.pt, 4, 4)
            return self.motion_unpatch(x)
        raise ValueError("indicator0=action,1=flow")


class LiberoFlowActionModel(LiberoActionModel):
    def configure_flow(self, codec, tokens=32, decoded_3d_weight=0.0):
        self.flow_codec = codec.to(device=self.device, dtype=self.torch_dtype)
        self.flow_adapters = FlowActionAdapters(self.action_expert.hidden_dim, int(tokens),
            motion_channels=codec.latent_channels).to(
            device=self.device, dtype=self.torch_dtype)
        self.decoded_3d_weight = float(decoded_3d_weight)
        self.last_shapes = {}
        self.configure_trainability()
        return self

    def configure_trainability(self):
        self.eval().requires_grad_(False)
        self.mot.train()
        self.video_expert.eval()
        self.action_expert.train().requires_grad_(True)
        # Original per-token linear head is retained for checkpoint compatibility,
        # but is unused and never optimized in this experiment.
        self.action_expert.head.requires_grad_(False)
        self.proprio_encoder.train().requires_grad_(True)
        self.flow_adapters.train().requires_grad_(True)
        self.flow_codec.eval().requires_grad_(False)

    def _prepare_policy(self, state, indicator, timestep, context, context_mask):
        expert, adapters = self.action_expert, self.flow_adapters
        if indicator == 0:
            tokens = expert.action_encoder(state).repeat_interleave(adapters.ratio, dim=1)
        else:
            tokens = adapters.embed_motion(state)
        ids = torch.full((state.shape[0],), indicator, device=state.device, dtype=torch.long)
        task = adapters.indicator(ids)
        time = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, timestep).to(expert.time_embedding[0].weight))
        time = time + task
        n = adapters.tokens
        return dict(tokens=tokens + adapters.position + task[:, None],
                    freqs=expert.freqs[:n].view(n, 1, -1).to(state.device),
                    t_mod=expert.time_projection(time).unflatten(1, (6, expert.hidden_dim)),
                    context=expert.text_embedding(context),
                    context_mask=context_mask[:, None, :].expand(-1, n, -1))

    def _conditions(self, sample, ids):
        video = sample["video"][ids].to(self.device, self.torch_dtype)
        context = sample["context"][ids].to(self.device, self.torch_dtype)
        mask = sample["context_mask"][ids].to(self.device, torch.bool)
        proprio = sample["proprio"][ids, 0].to(self.device, self.torch_dtype)
        context, mask = self._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
        with torch.no_grad():
            condition = self._encode_video_latents(video[:, :, :1], tiled=False)
            clip = self._encode_clip_fea_from_video(video[:, :, :1])
        logical_t = sample["video_num_frames"][ids]
        if not bool((logical_t == 17).all()):
            raise ValueError("Baseline logical video length must stay17")
        latent_t = (17 - 1) // int(self.vae.temporal_downsample_factor) + 1
        video_noise = torch.randn((len(ids), condition.shape[1], latent_t, *condition.shape[-2:]),
                                  device=self.device, dtype=self.torch_dtype)
        video_time = torch.full((len(ids),), float(self.train_video_scheduler.num_train_timesteps),
                                device=self.device, dtype=self.torch_dtype)
        cache, attention, video_len = self._build_frozen_video_cache(
            latents=video_noise, timestep_video=video_time, context=context, context_mask=mask,
            condition_latents=condition, clip_fea=clip, action_seq_len=self.flow_adapters.tokens,
            fuse_vae_embedding_in_latents=False)
        return context, mask, cache, attention, video_len

    def _velocity(self, state, indicator, timestep, conditions):
        context, mask, cache, attention, video_len = conditions
        pre = self._prepare_policy(state, indicator, timestep, context, mask)
        hidden = self.mot.forward_action_with_video_cache(
            action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
            action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache=cache, attention_mask=attention, video_seq_len=video_len)
        output = self.flow_adapters.output(hidden, indicator)
        if indicator == 1:
            # Bounded Gaussian velocity preconditioning retains the full noisy
            # latent around the lossy patch bottleneck. No GT enters this skip.
            sigma = (timestep.float() / self.train_action_scheduler.num_train_timesteps).view(-1, 1, 1, 1, 1)
            var = sigma.square() + (1-sigma).square()
            output = ((2*sigma-1)/var * state.float() + var.rsqrt() * output.float()).to(state.dtype)
        self.last_shapes[str(indicator)] = dict(indicator=indicator, noisy_state=list(state.shape),
            shared_tokens=list(pre["tokens"].shape), hidden=list(hidden.shape),
            velocity=list(output.shape), patch_thw=[self.flow_adapters.pt, 7, 7],
            action_downsample=self.flow_adapters.ratio, padding=0,
            route="action_head" if indicator == 0 else "ConvTranspose3d -> latent update -> MotionVAE decode")
        return output

    @staticmethod
    def _masked_mean(values, valid):
        valid = valid.expand_as(values).float()
        dims = tuple(range(1, values.ndim))
        return (values.float()*valid).sum(dims) / valid.sum(dims).clamp_min(1)

    def training_loss(self, sample, **kwargs):
        indicators = sample["indicator"]
        batch_size = len(indicators)
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        metrics = dict(loss_action=0.0, loss_flow=0.0, loss_3d=0.0,
                       fraction_action=float((indicators == 0).float().mean()))
        for indicator, name in ((0, "action"), (1, "flow")):
            ids = sample[f"{name}_indices"]
            if not len(ids):
                continue
            if not bool((indicators[ids] == indicator).all()):
                raise ValueError("Indicator/packed label mismatch")
            if indicator == 0:
                target = sample["action"].to(self.device, self.torch_dtype)
                valid = (~sample["action_is_pad"]).to(self.device)[:, :, None]
            else:
                maps = sample["motion"].to(self.device, self.torch_dtype)
                target = self.flow_codec.encode(maps)
                valid_map = sample["motion_valid"].to(self.device)
                # Suppress latent cells with invalid pixels in their8x8 image
                # footprint. This is an approximate latent mask: the VAE's
                # receptive field extends beyond one such footprint.
                b, views = maps.shape[:2]
                invalid = (~valid_map).float().reshape(b*views*2*8, 1, 224, 224)
                pooled = F.max_pool2d(invalid, 8, 8).reshape(b, views, 2, 8, 28, 28)
                valid = (pooled == 0).repeat_interleave(4, dim=2).reshape(b, 8*views, 8, 28, 28)
            noise = torch.randn_like(target)
            time = self.train_action_scheduler.sample_training_t(len(ids), self.device, target.dtype)
            noisy = self.train_action_scheduler.add_noise(target, noise, time)
            conditions = self._conditions(sample, ids)
            velocity = self._velocity(noisy, indicator, time, conditions)
            errors = (velocity.float() - (noise-target).float()).square()
            weight = self.train_action_scheduler.training_weight(time).float()
            task_loss = (self._masked_mean(errors, valid)*weight).sum()/batch_size
            loss = loss + task_loss
            metrics[f"loss_{name}"] = float(task_loss.detach())
            if indicator and self.decoded_3d_weight:
                sigma = (time.float()/self.train_action_scheduler.num_train_timesteps).view(-1,1,1,1,1)
                clean = (noisy.float()-sigma*velocity.float()).to(target.dtype)
                decoded = self.flow_codec.decode(clean)
                mask3d = valid_map.repeat_interleave(3, dim=2)
                l3d = self._masked_mean((decoded.float()-maps.float()).abs(), mask3d)
                l3d = (l3d*(1-sigma.flatten()).square()).sum()/batch_size*self.decoded_3d_weight
                loss = loss + l3d
                metrics["loss_3d"] = float(l3d.detach())
        # Task-specific modules may be unused on a rank/microbatch. Explicit zero
        # graph edges let distributed gradient reduction combine disjoint labels.
        for module in (self.flow_adapters, self.action_expert.action_encoder):
            for parameter in module.parameters():
                loss = loss + parameter.reshape(-1)[0].float()*0
        return loss, metrics

    @torch.no_grad()
    def sample_task(self, sample, indicator=0, steps=10):
        if indicator not in (0, 1) or steps != 10:
            raise ValueError("Use indicator0/1 and matched10-step sampling")
        ids = torch.arange(sample["video"].shape[0], device=sample["video"].device)
        conditions = self._conditions(sample, ids)
        shape = (len(ids), 32, 7) if indicator == 0 else (len(ids), self.flow_codec.latent_channels, 8, 28, 28)
        state = torch.randn(shape, device=self.device, dtype=self.torch_dtype)
        times, deltas = self.infer_action_scheduler.build_inference_schedule(steps, self.device, self.torch_dtype)
        for time, delta in zip(times, deltas):
            velocity = self._velocity(state, indicator, time.expand(len(ids)), conditions)
            state = self.infer_action_scheduler.step(velocity, delta, state)
        if indicator == 0:
            return {"action": state}
        maps = self.flow_codec.decode(state)
        self.last_shapes["1"]["decoded_3d"] = list(maps.shape)
        return {"motion_latent": state, "point_map": maps[:, :, :3], "scene_flow": maps[:, :, 3:]}

    def infer_action(self, *args, **kwargs):
        # The inherited sampler calls the original frozen linear action head.
        # Fail explicitly instead of silently evaluating a different policy.
        raise RuntimeError("Use sample_task(batch, indicator=0, steps=10) for this model; "
                           "the baseline infer_action sampler does not support the shared flow/action head")

    def save_flow_checkpoint(self, path, step=None, resolved_cfg=None):
        cfg = OmegaConf.to_container(resolved_cfg, resolve=True) if OmegaConf.is_config(resolved_cfg) else resolved_cfg
        torch.save(dict(format="vpp2.libero_flow.v1", step=step,
                        motion_cameras=list(self.flow_codec.motion_cameras),
                        action_expert=self.action_expert.state_dict(),
                        proprio_encoder=self.proprio_encoder.state_dict(),
                        flow_adapters=self.flow_adapters.state_dict(), resolved_config=cfg), path)

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("format") != "vpp2.libero_flow.v1":
            raise ValueError("Expected a dedicated flow checkpoint; use baseline action_init only through the factory")
        check_checkpoint_views(payload, self.flow_codec.motion_cameras)
        for name in ("action_expert", "proprio_encoder", "flow_adapters"):
            getattr(self, name).load_state_dict(payload[name], strict=True)
        return payload.get("step")
