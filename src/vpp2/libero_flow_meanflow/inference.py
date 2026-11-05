"""LIBERO deployment adapter for the shared flow/action training model.

Only observation packaging and deterministic evaluation RNG differ from
sample_task. The task embedding, velocity network, action head and scheduler
are inherited unchanged. No training or baseline module is modified.
"""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .model import LiberoFlowMeanFlow
from .model import LiberoReconMeanFlow


def training_observation_input(obs, cfg, processor, width, height, device, dtype):
    """Run the actual training processor, including FP32 image normalization."""
    from vpp2.libero.libero_utils import get_libero_image
    from vpp2.libero.eval_libero_single import _extract_sim_state
    if cfg.data.train.concat_multi_camera != 'horizontal':
        raise ValueError('Expected training horizontal camera layout')
    views = get_libero_image(obs)
    raw = dict(idx=0, task='',
               state={'default': torch.from_numpy(_extract_sim_state(obs))[None]},
               state_is_pad=torch.zeros(1, dtype=torch.bool),
               image_is_pad=torch.zeros(1, dtype=torch.bool),
               images={key: torch.from_numpy(value).permute(2, 0, 1)[None] for key, value in views.items()})
    item = processor.preprocess(raw)
    video = torch.cat(list(item['pixel_values'].unbind(0)), dim=-1)
    video = ((video - 0.5) / 0.5).permute(1, 0, 2, 3)
    image = video[:, 0][None].to(device=device, dtype=dtype)
    assert image.shape == (1, 3, height, width)
    return image, item['proprio'][:1], views


class FlowInferenceMixin:
    def load_checkpoint(self, path, optimizer=None, video_checkpoint_override=None, **kwargs):
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        trained = payload['resolved_config']
        if OmegaConf.to_container(self.training_flow_config, resolve=True) != trained['flow']:
            raise ValueError('Deployment flow settings differ from checkpoint')
        video = Path(trained['model']['dit_checkpoint_path']).resolve()
        if video_checkpoint_override and Path(video_checkpoint_override).resolve() != video:
            raise ValueError('Deployment frozen Video checkpoint differs from training')
        self._load_video_base_state_dict(self._load_video_checkpoint_state(video))
        step = super().load_checkpoint(path, optimizer=optimizer, **kwargs)
        self.eval().requires_grad_(False)
        modules = ['action_expert', 'proprio_encoder', 'flow_adapters']
        if self.training_flow_config.objective == 'direct_reconstruction':
            modules.append('flow_codec')
            if not torch.equal(self.motion_queries.detach().cpu(),
                               payload['motion_queries'].to(dtype=self.motion_queries.dtype)):
                raise ValueError('Checkpoint load mismatch: motion_queries')
        modules += [n for n in ('aux_action_head','aux_motion_unpatch') if hasattr(self,n)]
        for name in modules:
            actual = getattr(self, name).state_dict()
            for key, saved in payload[name].items():
                if not torch.equal(actual[key].detach().cpu(), saved.to(dtype=actual[key].dtype)):
                    raise ValueError(f'Checkpoint load mismatch: {name}.{key}')
        self._checkpoint_audit = dict(checkpoint=str(Path(path).resolve()), step=step,
                                     strict_load=True, all_saved_tensors_equal=True,
                                     restored_modules=modules,
                                     flow_objective=self.training_flow_config.objective,
                                     training_model_class=self.training_model_class.__name__,
                                     video_checkpoint=str(video))
        return step

    def _eval_conditions(self, sample, video_noise):
        # Same operations and order as LiberoFlowMeanFlow._conditions;
        # video_noise is supplied by the benchmark's independent CPU generator.
        video = sample['video'].to(self.device, self.torch_dtype)
        context = sample['context'].to(self.device, self.torch_dtype)
        mask = sample['context_mask'].to(self.device, torch.bool)
        proprio = sample['proprio'][:, 0].to(self.device, self.torch_dtype)
        context, mask = self._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
        condition = self._encode_video_latents(video[:, :, :1], tiled=False)
        clip = self._encode_clip_fea_from_video(video[:, :, :1])
        video_time = torch.full((len(video),), float(self.train_video_scheduler.num_train_timesteps),
                                device=self.device, dtype=self.torch_dtype)
        cache, attention, video_len = self._build_frozen_video_cache(
            latents=video_noise, timestep_video=video_time, context=context, context_mask=mask,
            condition_latents=condition, clip_fea=clip, action_seq_len=self.flow_adapters.tokens,
            fuse_vae_embedding_in_latents=False)
        return context, mask, cache, attention, video_len

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     context=None, context_mask=None, negative_prompt='', text_cfg_scale=1.,
                     num_inference_steps=1, sigma_shift=5., seed=42, video_seed=43,
                     rand_device='cpu', tiled=False, num_video_frames=17):
        if not hasattr(self, '_checkpoint_audit'):
            raise RuntimeError('Strict checkpoint restoration must finish before inference')
        if (action_horizon, num_inference_steps, num_video_frames) != (32, 1, 17):
            raise ValueError('Shared policy requires action32, NFE1, logical video17')
        if tiled or text_cfg_scale != 1 or negative_prompt or sigma_shift != 5:
            raise ValueError('Use the matched LIBERO evaluation settings')
        if seed is None or video_seed is None or rand_device != 'cpu':
            raise ValueError('Reference protocol requires independent fixed CPU RNGs')
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image[None]
        if tuple(input_image.shape) != (1, 3, 224, 448):
            raise ValueError(f'Expected horizontal dual-view RGB: {tuple(input_image.shape)}')
        if proprio is None or tuple(proprio.shape) != (1, 8):
            raise ValueError('Expected normalized proprio [1,8]')
        if context is None:
            context, context_mask = self.encode_prompt(prompt)
        elif prompt is not None or context_mask is None:
            raise ValueError('Pass prompt or complete cached context, not both')
        sample = dict(video=input_image[:, :, None].to(self.device, self.torch_dtype),
                      context=context.to(self.device, self.torch_dtype),
                      context_mask=context_mask.to(self.device, torch.bool),
                      proprio=proprio[:, None].to(self.device, self.torch_dtype),
                      video_num_frames=torch.tensor([17], device=self.device))
        def noise(shape, value):
            return torch.randn(shape, generator=torch.Generator(device='cpu').manual_seed(value),
                               device='cpu', dtype=torch.float32).to(self.device, self.torch_dtype)
        # Wan video VAE: C=16, temporal factor4, spatial factor8.
        video_noise = noise((1, 16, 5, 28, 56), video_seed)
        action_noise = noise((1, 32, 7), seed)
        conditions = self._eval_conditions(sample, video_noise)
        state = action_noise.clone()
        time = torch.ones(1, device=self.device, dtype=torch.float32)
        state = (state.float()-self._mean_velocity(state,0,torch.zeros_like(time),time,conditions).float()).to(state)
        if not torch.isfinite(state).all():
            raise ValueError('Nonfinite predicted action')
        del conditions
        if not getattr(self, '_inference_verified', False):
            # Execute the unmodified training diagnostic sampler on the first
            # real simulator observation with EXACTLY the same two noise draws.
            from unittest.mock import patch
            draws = [video_noise, action_noise]
            def fixed_randn(shape, *args, **kwargs):
                value = draws.pop(0)
                if tuple(shape) != tuple(value.shape):
                    raise ValueError(f'Unexpected reference noise shape: {shape}')
                return value.clone()
            with patch('torch.randn', side_effect=fixed_randn):
                reference = self.training_model_class.sample_task(self, sample, indicator=0, steps=1)['action']
            assert not draws
            error = float((reference.float() - state.float()).abs().max())
            if not torch.equal(reference, state):
                raise ValueError(f'Deployment differs from training sample_task: max_abs={error}')
            report = dict(self._checkpoint_audit, status='PASS', max_abs_vs_training_sample_task=error,
                          action_shape=list(state.shape), shapes=self.last_shapes['0'],
                          action_nfe=1, indicator=0, action_seed=seed, video_seed=video_seed,
                          input_video=list(sample['video'].shape), context_shape=list(context.shape),
                          mask_shape=list(context_mask.shape), proprio_shape=list(proprio.shape),
                          velocity_method='LiberoFlowMeanFlow._mean_velocity',
                          normalization='training processor and dataset_stats.json',
                          no_flow_label_at_inference=True)
            self.inference_audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.inference_audit_path.write_text(json.dumps(report, indent=2) + '\n')
            print('FLOW_INFERENCE_PARITY ' + json.dumps(report), flush=True)
            self._inference_verified = True
        return {'action': state}


class LiberoFlowInferenceModel(FlowInferenceMixin, LiberoFlowMeanFlow):
    training_model_class = LiberoFlowMeanFlow


class LiberoReconstructionInferenceModel(FlowInferenceMixin, LiberoReconMeanFlow):
    training_model_class = LiberoReconMeanFlow


def create_flow_inference_model(flow_config, meanflow_config, inference_audit_path, **kwargs):
    from vpp2.libero.runtime import create_action_model
    from vpp2.libero_flow.codec import MotionCodec
    cfg = OmegaConf.create(flow_config)
    if cfg.objective not in ('latent_meanflow', 'direct_reconstruction') or cfg.tokens not in (32, 64, 128):
        raise ValueError('Expected a 32/64/128-token shared FM or reconstruction model')
    codec_class, model_class, extra = MotionCodec, LiberoFlowInferenceModel, {}
    if cfg.objective == 'direct_reconstruction':
        from vpp2.libero_flow.reconstruction_codec import ReconstructionCodec
        codec_class, model_class = ReconstructionCodec, LiberoReconstructionInferenceModel
        extra = dict(reconstruction_loss=cfg.reconstruction_loss,
                     scene_flow_loss_weight=cfg.get('scene_flow_loss_weight', 1.0))
    print('FLOW_EVAL_LOADING_CODEC', flush=True)
    codec = codec_class(cfg.codec_path, frame_batch=cfg.codec_frame_batch, cameras=cfg.motion_cameras)
    # All learned components are replaced by strict pretrained/checkpoint loads.
    # Avoid spending minutes filling temporary Linear/Conv weights with random
    # values before replacing them. Constants and positional buffers are kept.
    # This context is local to deployment; training initialization is unchanged.
    from unittest.mock import patch
    if not kwargs.get('skip_dit_load_from_pretrain'):
        raise ValueError('Deployment expects explicit checkpoint restoration')
    print('FLOW_EVAL_LOADING_BACKBONE', flush=True)
    with patch('torch.nn.init.kaiming_uniform_', side_effect=lambda tensor, *a, **kw: tensor), \
         patch('torch.nn.init.uniform_', side_effect=lambda tensor, *a, **kw: tensor):
        model = create_action_model(_model_class=model_class, **kwargs)
    print('FLOW_EVAL_BACKBONE_CONSTRUCTED', flush=True)
    model.configure_flow(codec, cfg.tokens, cfg.decoded_3d_weight, **extra)
    model.configure_meanflow(meanflow_config)
    model.training_flow_config = cfg
    model.inference_audit_path = Path(inference_audit_path)
    # Match the training processor's FP32 normalization before BF16 casting.
    # This patch is confined to the dedicated flow evaluator worker process.
    from vpp2.libero import eval_libero_single as simulator
    simulator._obs_to_model_input = training_observation_input
    return model.eval().requires_grad_(False)
