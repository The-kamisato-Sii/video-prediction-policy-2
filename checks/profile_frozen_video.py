"""RoboDojo-shaped Action2B capacity probe with a frozen history-10k Video.

Reuses the released action-only model/trainer from LIBERO, with RoboDojo's
16-D actions/proprio, history, resolution and full five-frame latent context.
This is a synthetic capacity experiment, not the joint RoboDojo baseline.
"""

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from fullsize_meanflow import SyntheticRGB
from vpp2.cli import load_config
from vpp2.libero.trainer import LiberoTrainer


def digest(module):
    checksum = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        checksum.update(name.encode())
        checksum.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return checksum.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--search-max', action='store_true', help='Double then bisect; recover synchronized forward OOMs')
    parser.add_argument('--lower-bound', type=int, default=0, help='Previously verified batch on the same configuration')
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.steps < 2 or args.accumulation < 1:
        parser.error('Positive batch/accumulation and at least two optimizer steps required')
    if args.search_max and args.accumulation != 1:
        parser.error('Capacity search requires accumulation=1; validate accumulation separately')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.manual_seed(42)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'scope.json').exists():
        raise FileExistsError('Use a fresh output directory')
    cfg = load_config('configs/robodojo/train_8h200.yaml')
    for key in ('wan', 'video_init', 'action_init'):
        cfg.paths[key] = str(Path(cfg.paths[key]).resolve())
    cfg.output_dir = str(out)
    cfg.model._target_ = 'vpp2.libero.runtime.create_action_model'
    cfg.train_mode = 'action_only'
    cfg.checkpoint_mode = 'action_only'
    cfg.joint_learning_rates = None
    cfg.frozen_video_checkpoint = str(Path(cfg.paths.video_init).resolve())
    cfg.batch_size = args.batch_sizes[0]
    cfg.gradient_accumulation_steps = args.accumulation
    cfg.num_workers = 0
    cfg.warmup_steps = 0
    cfg.lr_tail = None
    cfg.max_steps = 100000
    cfg.save_final_checkpoint = False
    cfg.eval_every = 0
    cfg.eval_at_start = False
    cfg.preflight.enabled = False
    cfg.profiling.enabled = False
    cfg.wandb.enabled = False
    if rank == 0:
        (out / 'resolved.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=f'cuda:{torch.cuda.current_device()}')
    dataset = SyntheticRGB(max(world * max(args.batch_sizes) * args.accumulation * args.steps * 2, 64), 42 + rank)
    trainer = LiberoTrainer(model, dataset, cfg=cfg)
    trainer._set_dit_only_train_mode()
    engine = trainer.model
    assert trainer.accelerator.num_processes == world
    assert trainer._deepspeed_zero_stage() == 2
    assert not any(p.requires_grad for p in model.video_expert.parameters())
    allowed = {id(p) for m in (model.action_expert, model.proprio_encoder) for p in m.parameters()}
    assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
    assert all(p.grad is None for p in model.video_expert.parameters())
    frozen_before = digest(model.video_expert) if rank == 0 else None
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        import deepspeed
        write_json(out / 'scope.json', dict(
            scope='Synthetic RGB capacity; frozen Video, train Action2B plus proprio projection',
            video_checkpoint=cfg.frozen_video_checkpoint,
            action_checkpoint=str(Path(cfg.paths.action_init).resolve()),
            source_video_sha256='2d2490bd60272ec37423dabafaed4a889b2b2bf3760ac96d63d551d723ad2852',
            video_initial_tensor_sha256=frozen_before,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_video_parameters=sum(p.numel() for p in model.video_expert.parameters()),
            world_size=world, accumulation=args.accumulation, zero_stage=2,
            torch_version=torch.__version__, deepspeed_version=deepspeed.__version__,
            gpu=torch.cuda.get_device_name(), total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            video_shape=[3,17,240,416], condition_video_shape=[3,9,240,416], action_shape=[32,16],
            action_visible_latent_frames=5, first_frame_fast_path=False,
            learning_rate=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),
            note='Batch sizes reuse optimizer state; each batch measures complete updates, excludes loading and checksums.',
        ))
    grad_observations = {}
    for name, param in [('action', model.action_expert.head.weight), ('proprio', model.proprio_encoder.weight)]:
        def hook(grad, name=name):
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f'{name} gradient is non-finite')
            grad_observations[name] = float(grad.float().norm())
        param.register_hook(hook)
    pending = list(args.batch_sizes)
    lower, upper = args.lower_bound, None
    while pending:
        batch = pending.pop(0)
        if rank == 0:
            write_json(out / 'current_batch.json', dict(batch_size=batch, status='RUNNING'))
        engine.set_train_micro_batch_size(batch)
        assert engine.train_micro_batch_size_per_gpu() == batch
        assert engine.gradient_accumulation_steps() == args.accumulation
        assert engine.train_batch_size() == batch * world * args.accumulation
        sample = default_collate([dataset[0] for _ in range(batch)])
        initial_action = model.action_expert.head.weight.detach().clone()
        initial_proprio = model.proprio_encoder.weight.detach().clone()
        initial_engine_steps = engine.global_steps
        grad_observations.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        capacity_oom = False
        for step in range(args.steps):
            trainer.accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for micro in range(args.accumulation):
                with trainer.accelerator.accumulate(engine):
                    loss, metrics, oom_message = None, None, None
                    try:
                        with trainer.accelerator.autocast():
                            loss, metrics = engine(sample)
                    except torch.cuda.OutOfMemoryError as error:
                        oom_message = str(error)
                    oom_flag = torch.tensor(int(oom_message is not None), device='cuda')
                    dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                    if oom_flag.item():
                        # All ranks agree before any backward collective is entered.
                        messages = [None] * world
                        dist.all_gather_object(messages, oom_message)
                        if rank == 0:
                            write_json(out / f'batch_{batch}_OOM.json', dict(batch_size=batch, phase='forward', errors=messages))
                            print(f'CAPACITY OOM batch={batch}', flush=True)
                        del loss, metrics
                        capacity_oom = True
                        break
                    assert torch.isfinite(loss), metrics
                    trainer.accelerator.backward(loss)
                    if trainer.accelerator.sync_gradients:
                        trainer.accelerator.clip_grad_norm_(engine.parameters(), trainer.max_grad_norm)
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad(set_to_none=True)
            if capacity_oom:
                break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            assert engine.global_steps == initial_engine_steps + step + 1
            if rank == 0:
                print(json.dumps(dict(batch=batch, step=step+1, seconds=elapsed, loss=float(loss.detach()))), flush=True)
            del loss, metrics
        if capacity_oom:
            del sample, initial_action, initial_proprio
            engine.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            upper = batch if upper is None else min(upper, batch)
            if args.search_max and upper - lower > 1:
                pending = [(upper + lower) // 2]
            else:
                pending = []
            continue
        assert all(p.grad is None for p in model.video_expert.parameters())
        assert all(grad_observations.get(k, 0) > 0 for k in ('action', 'proprio'))
        assert not torch.equal(initial_action, model.action_expert.head.weight)
        assert not torch.equal(initial_proprio, model.proprio_encoder.weight)
        local = dict(rank=rank, batch_size=batch, accumulation=args.accumulation,
                     seconds=timings, steady_mean_seconds=sum(timings[1:])/(len(timings)-1),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                     observed_grad_norms=grad_observations.copy(), action_updated=True, proprio_updated=True,
                     frozen_video_grad_none=True, engine_optimizer_steps=args.steps)
        # Full Action checksum on every rank confirms post-update replicas agree.
        action_hash = digest(model.action_expert)
        hashes = [None] * world
        dist.all_gather_object(hashes, action_hash)
        assert len(set(hashes)) == 1, 'Action replicas diverged'
        local['action_replicas_identical'] = True
        write_json(out / f'batch_{batch}_rank{rank}.json', local)
        if rank == 0:
            write_json(out / f'batch_{batch}_PASS.json', local)
            print(f'PASS batch={batch}: Video gradients absent, Action/proprio updated, all Action replicas identical', flush=True)
        del sample, initial_action, initial_proprio
        lower = max(lower, batch)
        if args.search_max:
            pending = [batch * 2] if upper is None else ([(upper + lower) // 2] if upper - lower > 1 else [])
    if rank == 0:
        assert digest(model.video_expert) == frozen_before, 'Frozen Video changed'
        write_json(out / 'capacity_summary.json', dict(max_passed_batch=lower, min_forward_oom_batch=upper,
                   accumulation=args.accumulation, world_size=world, video_full_checksum_unchanged_rank0=True))
        (out / 'PASS').write_text('PASS\n')
    trainer.accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
