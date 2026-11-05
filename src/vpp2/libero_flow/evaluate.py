"""Dedicated flow-checkpoint coordinator using the unchanged LIBERO harness."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from vpp2.libero import evaluate as harness

_baseline_worker_config = harness.worker_config


def validate_checkpoint(cfg):
    from vpp2.libero.cli import validate_training_config
    from vpp2.libero.protocols import validate_config
    p = validate_config(cfg)
    action = torch.load(cfg.action_checkpoint, map_location='cpu', mmap=True, weights_only=True)
    training = OmegaConf.load(cfg.training_config)
    assert action['resolved_config'] == OmegaConf.to_container(training, resolve=True)
    validate_training_config(training, require_portable_target=False)
    f = training.flow
    expected_format = {'latent_flow_matching': 'vpp2.libero_flow.v1',
                       'direct_reconstruction': 'vpp2.libero_flow.reconstruction.v1'}
    if action.get('format') != expected_format.get(f.objective) or action.get('step') != 30000:
        raise ValueError('Expected a final shared flow/action checkpoint matching its training objective')
    assert int(cfg.action_step) == 30000 and int(cfg.video_step) == 10000
    assert f.tokens in (32, 64, 128) and f.action_denominator in (1, 2, 4, 8)
    tokens = int(f.tokens)
    temporal_patch, action_downsample = 128 // tokens, tokens // 32
    assert f.dt == 4
    assert list(f.motion_cameras) == ['image'] and action['motion_cameras'] == ['image']
    assert f.motion_normalization == 'selected_views_clip_mean_radius_v1'
    assert int(cfg.action_nfe) == 10 and cfg.action_sampler == 'flow_matching'
    assert cfg.benchmark == 'pro' and cfg.protocol == 'public800' and p['episodes'] == 800
    assert Path(cfg.video_checkpoint).resolve() == Path(training.model.dit_checkpoint_path).resolve()
    assert Path(cfg.dataset_stats).resolve() == Path(training.data.train.pretrained_norm_stats).resolve()
    assert Path(cfg.wan_model_dir).resolve() == Path(training.model.model_id).resolve()
    assert tuple(action['flow_adapters']['position'].shape) == (1, tokens, 512)
    assert tuple(action['flow_adapters']['motion_patch.weight'].shape) == (512, 8, temporal_patch, 7, 7)
    assert tuple(action['flow_adapters']['action_head.0.weight'].shape) == (128, 512, action_downsample)
    modules = ['action_expert', 'proprio_encoder', 'flow_adapters']
    if f.objective == 'direct_reconstruction':
        assert action['reconstruction_loss'] == f.reconstruction_loss
        assert float(action.get('scene_flow_loss_weight', 1.0)) == float(f.get('scene_flow_loss_weight', 1.0))
        assert tuple(action['motion_queries'].shape) == (1, tokens, 512)
        assert torch.isfinite(action['motion_queries']).all()
        modules.append('flow_codec')
    for module in modules:
        assert all(torch.isfinite(v).all() for v in action[module].values()), module
    report = dict(status='PASS', checkpoint=str(cfg.action_checkpoint), action_step=30000,
                  tokens=tokens, action_downsample=action_downsample,
                  action_denominator=f.action_denominator, action_nfe=10,
                  indicator=0, protocol=p, training_config_matches_checkpoint=True,
                  flow_objective=f.objective, checkpoint_format=action['format'],
                  flow_normalization=f.motion_normalization, all_weights_finite=True)
    dest = Path(cfg.output_dir).parent / 'checkpoint-check.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def worker_config(cfg, output, manifest, gpu, seed_id):
    c = _baseline_worker_config(cfg, output, manifest, gpu, seed_id)
    c.model._target_ = 'vpp2.libero_flow.inference.create_flow_inference_model'
    c.model.flow_config = OmegaConf.to_container(c.flow, resolve=True)
    c.model.inference_audit_path = str(Path(cfg.output_dir).parent / f'inference-parity-gpu{gpu}.json')
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.check_only:
        validate_checkpoint(cfg)
        return
    # In-process dispatch only: no edits to the shared baseline coordinator.
    harness.validate_checkpoint = validate_checkpoint
    harness.worker_config = worker_config
    harness.run(cfg, a.gpus, resume=a.resume)


if __name__ == '__main__':
    main()
