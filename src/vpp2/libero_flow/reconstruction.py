"""One-pass query reconstruction ablation; action flow matching stays unchanged."""
import math
import torch
from torch import nn
from omegaconf import OmegaConf

from vpp2.models.wan21_14b.wan_video_dit import sinusoidal_embedding_1d
from .model import LiberoFlowActionModel
from .views import check_checkpoint_views


class LiberoFlowReconstructionModel(LiberoFlowActionModel):
    flow_inference_steps = 1
    checkpoint_format = "vpp2.libero_flow.reconstruction.v1"

    def configure_flow(self, codec, tokens=32, decoded_3d_weight=0.0, reconstruction_loss="l1",
                       scene_flow_loss_weight=1.0):
        if decoded_3d_weight:
            raise ValueError("decoded_3d_weight belongs to latent FM, not direct reconstruction")
        if reconstruction_loss not in ("l1", "mse"):
            raise ValueError("reconstruction_loss must be l1 or mse")
        self.reconstruction_loss = reconstruction_loss
        self.scene_flow_loss_weight = float(scene_flow_loss_weight)
        if not math.isfinite(self.scene_flow_loss_weight) or self.scene_flow_loss_weight <= 0:
            raise ValueError("scene_flow_loss_weight must be finite and positive")
        super().configure_flow(codec, tokens, 0.0)
        # The only additional parameters. Keep subsequent training RNG unchanged.
        with torch.random.fork_rng(devices=[torch.device(self.device).index or 0]
                                   if torch.device(self.device).type == "cuda" else []):
            self.motion_queries = nn.Parameter(torch.randn(
                1, int(tokens), self.action_expert.hidden_dim,
                device=self.device, dtype=self.torch_dtype) * 0.02)
        self.configure_trainability()
        return self

    def configure_trainability(self):
        super().configure_trainability()
        # Queries replace the noisy-motion embedding input. Keep the unused
        # input convolution in the inventory, but do not optimize it.
        self.flow_adapters.motion_patch.requires_grad_(False)
        self.flow_codec.train()
        for name in self.flow_codec.decoder_names:
            getattr(self.flow_codec, name).requires_grad_(True)
        if hasattr(self, "motion_queries"):
            self.motion_queries.requires_grad_(True)

    def _reconstruct(self, batch_size, conditions):
        context, mask, cache, attention, video_len = conditions
        expert, adapters = self.action_expert, self.flow_adapters
        queries = self.motion_queries.expand(batch_size, -1, -1)
        ids = torch.ones(batch_size, device=self.device, dtype=torch.long)
        task = adapters.indicator(ids)
        # Existing time-conditioning modules remain; a fixed zero marks clean
        # reconstruction. No random motion, diffusion schedule or GT input.
        time = torch.zeros(batch_size, device=self.device, dtype=self.torch_dtype)
        time = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, time).to(
            expert.time_embedding[0].weight)) + task
        tokens = queries + adapters.position + task[:, None]
        hidden = self.mot.forward_action_with_video_cache(
            action_tokens=tokens,
            action_freqs=expert.freqs[:adapters.tokens].view(adapters.tokens, 1, -1).to(self.device),
            action_t_mod=expert.time_projection(time).unflatten(1, (6, expert.hidden_dim)),
            action_context_payload={"context": expert.text_embedding(context),
                                    "mask": mask[:, None, :].expand(-1, adapters.tokens, -1)},
            video_kv_cache=cache, attention_mask=attention, video_seq_len=video_len)
        latent = adapters.output(hidden, 1)
        decoded = self.flow_codec.decode(latent)
        self.last_shapes["1"] = dict(indicator=1, learned_queries=list(queries.shape),
            shared_tokens=list(tokens.shape), hidden=list(hidden.shape), clean_latent=list(latent.shape),
            decoded_3d=list(decoded.shape), patch_thw=[adapters.pt, 7, 7], padding=0,
            action_downsample=adapters.ratio, dit_evaluations=1,
            route="learnable queries -> DiT -> ConvTranspose3d -> random trainable VAE decoder")
        return latent, decoded

    def reconstruction_errors(self, decoded, maps, valid):
        # Return unweighted errors in normalized 3D units. Task weighting is
        # applied after each quantity's separate validity-mask normalization.
        losses = []
        for channel in (0, 1):
            error = decoded[:, :, 3*channel:3*(channel+1)].float() - maps[:, :, 3*channel:3*(channel+1)].float()
            error = error.abs() if self.reconstruction_loss == "l1" else error.square()
            losses.append(self._masked_mean(error, valid[:, :, channel:channel+1]))
        return losses

    def training_loss(self, sample, **kwargs):
        indicators = sample["indicator"]
        batch_size = len(indicators)
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        metrics = dict(loss_action=0.0, loss_flow=0.0, loss_point_map=0.0, loss_scene_flow=0.0,
                       loss_scene_flow_weighted=0.0,
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
                noise = torch.randn_like(target)
                time = self.train_action_scheduler.sample_training_t(len(ids), self.device, target.dtype)
                noisy = self.train_action_scheduler.add_noise(target, noise, time)
                conditions = self._conditions(sample, ids)
                velocity = self._velocity(noisy, 0, time, conditions)
                errors = (velocity.float() - (noise-target).float()).square()
                weight = self.train_action_scheduler.training_weight(time).float()
                task_loss = (self._masked_mean(errors, valid)*weight).sum()/batch_size
            else:
                conditions = self._conditions(sample, ids)
                _, decoded = self._reconstruct(len(ids), conditions)
                # GT is used exclusively AFTER prediction, in the loss.
                maps = sample["motion"].to(self.device, torch.float32)
                valid = sample["motion_valid"].to(self.device)
                geom, motion = self.reconstruction_errors(decoded, maps, valid)
                weighted_motion = self.scene_flow_loss_weight * motion
                task_loss = (0.5*(geom+weighted_motion)).sum()/batch_size
                metrics["loss_point_map"] = float(geom.detach().mean())
                metrics["loss_scene_flow"] = float(motion.detach().mean())
                metrics["loss_scene_flow_weighted"] = float(weighted_motion.detach().mean())
            loss = loss + task_loss
            metrics[f"loss_{name}"] = float(task_loss.detach())
        # Different ranks can have disjoint labels. Include all task-specific
        # trainable parameters so ZeRO/DDP can reduce both branches safely.
        modules = [self.flow_adapters, self.action_expert.action_encoder]
        modules += [getattr(self.flow_codec, name) for name in self.flow_codec.decoder_names]
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    loss = loss + parameter.reshape(-1)[0].float()*0
        loss = loss + self.motion_queries.reshape(-1)[0].float()*0
        return loss, metrics

    @torch.no_grad()
    def sample_task(self, sample, indicator=0, steps=None):
        if indicator == 0:
            return super().sample_task(sample, indicator=0, steps=10 if steps is None else steps)
        if indicator != 1 or steps not in (None, 1):
            raise ValueError("Reconstruction uses exactly one DiT evaluation; actions use10")
        ids = torch.arange(sample["video"].shape[0], device=sample["video"].device)
        latent, maps = self._reconstruct(len(ids), self._conditions(sample, ids))
        return {"motion_latent": latent, "point_map": maps[:, :, :3], "scene_flow": maps[:, :, 3:]}

    def save_flow_checkpoint(self, path, step=None, resolved_cfg=None):
        cfg = OmegaConf.to_container(resolved_cfg, resolve=True) if OmegaConf.is_config(resolved_cfg) else resolved_cfg
        torch.save(dict(format=self.checkpoint_format, step=step,
            motion_cameras=list(self.flow_codec.motion_cameras),
            action_expert=self.action_expert.state_dict(), proprio_encoder=self.proprio_encoder.state_dict(),
            flow_adapters=self.flow_adapters.state_dict(), motion_queries=self.motion_queries.detach(),
            # Include retained frozen random encoders for exact state restoration.
            flow_codec=self.flow_codec.state_dict(), reconstruction_loss=self.reconstruction_loss,
            scene_flow_loss_weight=self.scene_flow_loss_weight,
            resolved_config=cfg), path)

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload.get("format") != self.checkpoint_format:
            raise ValueError("Expected a direct-reconstruction checkpoint, not a latent-FM checkpoint")
        check_checkpoint_views(payload, self.flow_codec.motion_cameras)
        if payload["reconstruction_loss"] != self.reconstruction_loss:
            raise ValueError("Checkpoint reconstruction loss differs from configuration")
        if float(payload.get("scene_flow_loss_weight", 1.0)) != self.scene_flow_loss_weight:
            raise ValueError("Checkpoint scene-flow loss weight differs from configuration")
        if payload["motion_queries"].shape != self.motion_queries.shape:
            raise ValueError("Checkpoint query/token count differs from configuration")
        for name in ("action_expert", "proprio_encoder", "flow_adapters", "flow_codec"):
            getattr(self, name).load_state_dict(payload[name], strict=True)
        with torch.no_grad():
            self.motion_queries.copy_(payload["motion_queries"])
        return payload.get("step")
