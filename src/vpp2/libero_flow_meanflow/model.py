"""One-step shared action/motion policies using the existing MF/iMF objectives.

The loss receives [B, elements, 1] views so its original padding semantics mask
individual latent elements, without changing reductions or derivative formulas.
"""
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from vpp2.libero_flow.model import LiberoFlowActionModel
from vpp2.libero_flow.reconstruction import LiberoFlowReconstructionModel
from vpp2.libero_flow.views import check_checkpoint_views
from vpp2.meanflow.model import derivative_mode
from vpp2.meanflow.objectives import data_loss, improved_data_loss, directional_derivative, sample_intervals
from vpp2.meanflow.loss_config import loss_settings
from vpp2.meanflow.experiment_config import interval_settings, validate_interval_objective, imf_settings
from vpp2.models.wan21_14b.wan_video_dit import sinusoidal_embedding_1d


class SharedMeanFlowMixin:
    action_inference_steps = flow_inference_steps = 1
    checkpoint_format = 'vpp2.libero_flow.meanflow.v1'

    def configure_meanflow(self, config):
        cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else deepcopy(config)
        validate_interval_objective(cfg, 'libero')
        if cfg['objective'] not in ('data_meanflow', 'data_improved_meanflow'):
            raise ValueError('Only data MF and iMF are supported')
        self.meanflow_config = dict(cfg, **loss_settings(cfg))
        self.meanflow_config['interval'] = interval_settings(cfg, self.train_action_scheduler.shift)
        self.action_sampler = 'meanflow'
        # Match the action-only implementation; preserve subsequent RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.action_expert.enable_meanflow()
        self.auxiliary = cfg.get('imf', {}).get('variant') == 'auxiliary'
        if cfg['objective'] == 'data_improved_meanflow':
            self.meanflow_config['imf'] = imf_settings(cfg)
        if self.auxiliary:
            self.action_expert.enable_aux_v(8)
            # The original linear ActionDiT head is unused in flow+action.
            # v needs copies of the ACTUAL u output heads, not that unused head.
            self.aux_action_head = deepcopy(self.flow_adapters.action_head)
            if not self.reconstruct_flow:
                self.aux_motion_unpatch = deepcopy(self.flow_adapters.motion_unpatch)
        self.configure_trainability()
        return self

    def configure_trainability(self):
        super().configure_trainability()
        if hasattr(self.action_expert, 'aux_v_head'):
            self.action_expert.aux_v_head.requires_grad_(False)
        for name in ('aux_action_head', 'aux_motion_unpatch'):
            if hasattr(self, name):
                getattr(self, name).train().requires_grad_(True)

    @staticmethod
    def _slice_conditions(conditions, rows):
        context, mask, cache, attention, length = conditions
        # Baseline cache is K/V [B,...]; attention is a shared 2D mask.
        return (context.detach()[rows], mask[rows],
                [{k: v.detach()[rows] for k, v in layer.items()} for layer in cache],
                attention, length)

    def _mean_velocity(self, state, indicator, r, t, conditions, branch='u'):
        context, mask, cache, attention, video_len = conditions
        expert, adapters = self.action_expert, self.flow_adapters
        timestep = t * self.train_action_scheduler.num_train_timesteps
        pre = self._prepare_policy(state, indicator, timestep, context, mask)
        # Recompute combined t/r conditioning before the nonlinear projection.
        time = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, timestep).to(expert.time_embedding[0].weight))
        time = time + expert.time_end_embedding(sinusoidal_embedding_1d(
            expert.freq_dim, r * self.train_action_scheduler.num_train_timesteps).to(expert.time_end_embedding[0].weight))
        task = adapters.indicator(torch.full((len(state),), indicator, device=state.device, dtype=torch.long))
        hidden = self.mot.forward_action_with_video_cache(
            action_tokens=pre['tokens'], action_freqs=pre['freqs'],
            action_t_mod=expert.time_projection(time + task).unflatten(1, (6, expert.hidden_dim)),
            action_context_payload={'context': pre['context'], 'mask': pre['context_mask']},
            video_kv_cache=cache, attention_mask=attention, video_seq_len=video_len,
            action_branch=branch)
        if branch == 'u':
            output = adapters.output(hidden, indicator)
        elif indicator == 0:
            output = self.aux_action_head(hidden.transpose(1, 2)).transpose(1, 2)
        else:
            output = self.aux_motion_unpatch(hidden.transpose(1, 2).reshape(
                len(state), expert.hidden_dim, 8 // adapters.pt, 4, 4))
        if indicator == 1:
            sigma = t.float().view(-1, 1, 1, 1, 1)
            var = sigma.square() + (1-sigma).square()
            output = ((2*sigma-1)/var * state.float() + var.rsqrt()*output.float()).to(state.dtype)
        return output

    def _meanflow_loss(self, target, valid, indicator, conditions):
        cfg = self.meanflow_config
        shape = target.shape[1:]
        clean = target.reshape(len(target), -1, 1)
        pad = ~valid.expand_as(target).reshape(len(target), -1).bool()
        noise = torch.randn_like(clean)
        interval = cfg['interval']
        t, r = sample_intervals(len(target), target.device, self.train_action_scheduler.shift,
                               interval['diagonal_probability'], interval['diagonal_time_sampling'])
        def forward(z, lo, hi, cond, branch='u'):
            return self._mean_velocity(z.reshape(len(z), *shape), indicator, lo, hi, cond, branch).reshape_as(z)
        def predict(z, lo, hi):
            return forward(z, lo, hi, conditions)
        def derivative(z, lo, hi, direction, rows):
            # The detached JVP is sample-independent. Bound the temporary KV
            # copies without changing the live batch, loss reduction or gradients.
            values = []
            with derivative_mode(self):
                for selected in rows.nonzero(as_tuple=True)[0].split(4):
                    fixed = self._slice_conditions(conditions, selected)
                    values.append(directional_derivative(lambda x, a, b: forward(x, a, b, fixed),
                        z[selected], lo[selected], hi[selected], direction[selected], backend=cfg['derivative_backend']))
            return torch.cat(values, dim=0)
        options = dict(baseline_reduction=True, loss_weighting=cfg['loss_weighting'],
            adaptive_power=cfg['adaptive_weighting']['power'], adaptive_c=cfg['adaptive_weighting']['c'],
            distributed=self.action_expert.training)
        if cfg['objective'] == 'data_improved_meanflow':
            def predict_v(z, time, rows):
                if self.auxiliary:
                    return forward(z, time, time, conditions, 'aux_v')
                return torch.cat([forward(z[selected], time[selected], time[selected],
                    self._slice_conditions(conditions, selected))
                    for selected in rows.nonzero(as_tuple=True)[0].split(4)], dim=0)
            loss, metrics = improved_data_loss(predict, derivative, predict_v, clean, noise, r, t, pad,
                auxiliary=self.auxiliary, auxiliary_loss_weight=cfg['imf'].get('auxiliary_loss_weight', 1.0), **options)
        else:
            loss, metrics = data_loss(predict, derivative, clean, noise, r, t, pad, **options)
        self.last_shapes[str(indicator)] = dict(indicator=indicator, noisy_state=list(target.shape),
            shared_tokens=[len(target), self.flow_adapters.tokens, self.action_expert.hidden_dim],
            velocity=list(target.shape), patch_thw=[self.flow_adapters.pt,7,7], padding=0,
            action_downsample=self.flow_adapters.ratio, inference_steps=1)
        metrics.update(time_mean=t.mean(), gap_mean=(t-r).mean(), diagonal_fraction=(t==r).float().mean())
        return loss, metrics

    def training_loss(self, sample, **kwargs):
        indicators = sample['indicator']
        # Required by the existing sampler and rank-symmetric objective collectives.
        if not bool((indicators == indicators[0]).all()):
            raise ValueError('MF flow experiments require homogeneous_interleaved batches')
        indicator = int(indicators[0]); name = 'action' if indicator == 0 else 'flow'
        ids = sample[f'{name}_indices']
        if len(ids) != len(indicators):
            raise ValueError('Indicator/packed labels mismatch')
        conditions = self._conditions(sample, ids)
        if indicator == 1 and self.reconstruct_flow:
            _, decoded = self._reconstruct(len(ids), conditions)
            maps = sample['motion'].to(self.device, torch.float32)
            valid = sample['motion_valid'].to(self.device)
            geom, motion = self.reconstruction_errors(decoded, maps, valid)
            loss = (0.5*(geom+self.scene_flow_loss_weight*motion)).mean()
            metrics = dict(loss_point_map=geom.mean(), loss_scene_flow=motion.mean(),
                           loss_scene_flow_weighted=self.scene_flow_loss_weight*motion.mean())
        else:
            if indicator == 0:
                target = sample['action'].to(self.device, self.torch_dtype)
                valid = (~sample['action_is_pad']).to(self.device)[:, :, None]
            else:
                maps = sample['motion'].to(self.device, self.torch_dtype)
                target = self.flow_codec.encode(maps)
                valid_map = sample['motion_valid'].to(self.device)
                b, views = maps.shape[:2]
                invalid = (~valid_map).float().reshape(b*views*2*8, 1, 224, 224)
                pooled = F.max_pool2d(invalid, 8, 8).reshape(b, views, 2, 8, 28, 28)
                valid = (pooled == 0).repeat_interleave(4, dim=2).reshape(b, 8*views, 8, 28, 28)
            loss, metrics = self._meanflow_loss(target, valid, indicator, conditions)
        metrics.update(loss_action=loss.detach() if indicator==0 else 0.,
                       loss_flow=loss.detach() if indicator==1 else 0., fraction_action=float(indicator==0))
        # All task-specific trainable params participate in every backward. This
        # includes the auxiliary tail on reconstruction-only batches and r at r=0.
        for parameter in self.parameters():
            if parameter.requires_grad:
                loss = loss + parameter.reshape(-1)[0].float()*0
        return loss, {k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items()}

    @torch.no_grad()
    def sample_task(self, sample, indicator=0, steps=1):
        if indicator not in (0,1) or steps != 1:
            raise ValueError('Both branches use exactly one policy evaluation')
        ids = torch.arange(sample['video'].shape[0], device=sample['video'].device)
        conditions = self._conditions(sample, ids)
        if indicator == 1 and self.reconstruct_flow:
            latent, maps = self._reconstruct(len(ids), conditions)
        else:
            shape = (len(ids),32,7) if indicator==0 else (len(ids),self.flow_codec.latent_channels,8,28,28)
            noise = torch.randn(shape, device=self.device, dtype=self.torch_dtype)
            time = torch.ones(len(ids),device=self.device,dtype=torch.float32)
            latent = (noise.float()-self._mean_velocity(noise,indicator,torch.zeros_like(time),time,conditions).float()).to(noise)
            self.last_shapes[str(indicator)] = dict(indicator=indicator, noisy_state=list(shape),
                shared_tokens=[len(ids),self.flow_adapters.tokens,self.action_expert.hidden_dim],
                velocity=list(shape), inference_steps=1)
            if indicator==0:
                return {'action':latent}
            maps = self.flow_codec.decode(latent)
        return dict(motion_latent=latent,point_map=maps[:,:,:3],scene_flow=maps[:,:,3:])

    def infer_action(self, *args, **kwargs):
        raise RuntimeError('Use the dedicated MF deployment adapter or sample_task(..., steps=1)')

    def save_flow_checkpoint(self, path, step=None, resolved_cfg=None):
        cfg = OmegaConf.to_container(resolved_cfg, resolve=True) if OmegaConf.is_config(resolved_cfg) else resolved_cfg
        names = ['action_expert','proprio_encoder','flow_adapters']
        if self.reconstruct_flow:names.append('flow_codec')
        names += [n for n in ('aux_action_head','aux_motion_unpatch') if hasattr(self,n)]
        payload = dict(format=self.checkpoint_format, step=step, resolved_config=cfg,
            motion_cameras=list(self.flow_codec.motion_cameras), meanflow=self.meanflow_config,
            reconstruct_flow=self.reconstruct_flow, action_nfe=1, flow_nfe=1,
            **{n:getattr(self,n).state_dict() for n in names})
        if self.reconstruct_flow:
            payload.update(motion_queries=self.motion_queries.detach(),reconstruction_loss=self.reconstruction_loss,
                           scene_flow_loss_weight=self.scene_flow_loss_weight)
        torch.save(payload,path)

    def load_checkpoint(self,path,optimizer=None,**kwargs):
        p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if p.get('format')!=self.checkpoint_format or p['meanflow']!=self.meanflow_config or p['reconstruct_flow']!=self.reconstruct_flow:
            raise ValueError('Different MF/iMF architecture or objective; cannot load checkpoint')
        check_checkpoint_views(p,self.flow_codec.motion_cameras)
        names=['action_expert','proprio_encoder','flow_adapters']
        if self.reconstruct_flow:
            assert p['reconstruction_loss']==self.reconstruction_loss and p['scene_flow_loss_weight']==self.scene_flow_loss_weight
            names.append('flow_codec')
            with torch.no_grad():self.motion_queries.copy_(p['motion_queries'])
        names += [n for n in ('aux_action_head','aux_motion_unpatch') if hasattr(self,n)]
        for n in names:getattr(self,n).load_state_dict(p[n],strict=True)
        return p.get('step')


class LiberoFlowMeanFlow(SharedMeanFlowMixin, LiberoFlowActionModel):
    reconstruct_flow = False


class LiberoReconMeanFlow(SharedMeanFlowMixin, LiberoFlowReconstructionModel):
    reconstruct_flow = True
