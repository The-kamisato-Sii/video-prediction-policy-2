"""Explicit opt-in launcher. Dry runs do not allocate GPUs."""

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def load_config(path, overrides=()):
    def read(path, parents=()):
        path = Path(path).resolve()
        if path in parents:
            raise ValueError("Configuration inheritance cycle")
        cfg = OmegaConf.load(path)
        base = cfg.pop("_base_", None)
        return (
            cfg
            if base is None
            else OmegaConf.merge(read(path.parent / base, (*parents, path)), cfg)
        )

    cfg = OmegaConf.merge(read(path), OmegaConf.from_dotlist(overrides))
    m = cfg.meanflow
    if cfg.benchmark not in {"robodojo", "libero"}:
        raise ValueError("Unknown benchmark")
    if m.objective not in {"teacher_trajectory", "data_meanflow", "kmf", "direct_action"}:
        raise ValueError("Unsupported objective; hybrid must be a separate explicit experiment")
    if (
        m.time_convention != "clean0_noise1"
        or m.loss_weighting != "uniform"
        or m.auxiliary_geometry
    ):
        raise ValueError("Only clean0/noise1 uniform objectives without geometry are implemented")
    if not m.preserve_baseline_trainability:
        raise ValueError("A frozen-condition variant needs its own experiment implementation")
    if m.objective == "teacher_trajectory":
        if m.derivative_backend != "none" or m.interval.diagonal_probability != 0:
            raise ValueError("Teacher trajectory uses no JVP and strictly nonzero intervals")
        expected_shift = 1.0 if cfg.benchmark == "robodojo" else 5.0
        if m.teacher.steps != 10 or m.teacher.sigma_shift != expected_shift:
            raise ValueError("Main teacher must use the original 10-step deployment schedule")
    elif m.objective == "direct_action":
        if cfg.benchmark != "libero" or m.derivative_backend != "none":
            raise ValueError("Direct Action MSE is LIBERO-only and uses no derivative")
    elif m.derivative_backend not in {"jvp", "finite_difference"}:
        raise ValueError("Data objectives require an explicit derivative backend")
    if m.objective == "kmf" and m.split not in {"learned", "midpoint"}:
        raise ValueError("K-MF split must be learned or midpoint")
    if cfg.model.action_dit_config.num_layers != 40 or cfg.model.video_dit_config.num_layers != 40:
        raise ValueError("Both original 40-layer experts must be retained")
    expected_sampler = "direct_action" if m.objective == "direct_action" else "meanflow"
    if cfg.inference.action_sampler != expected_sampler or cfg.inference.action_nfe != 1:
        raise ValueError("MeanFlow deployment must use one action NFE")
    if cfg.resume_ckpt:
        raise ValueError(
            "Use meanflow.student_init for initialization or resume for training state"
        )
    if cfg.benchmark == "libero":
        from vpp2.libero.cli import validate_training_config

        validate_training_config(cfg, require_portable_target=False)
    else:
        if cfg.train_mode != "joint" or cfg.checkpoint_mode != "full":
            raise ValueError("RoboDojo must retain full joint training")
        a = cfg.model.action_dit_config
        if (a.hidden_dim, a.ffn_dim, a.num_heads, a.attn_head_dim) != (1024, 4096, 40, 128):
            raise ValueError("RoboDojo must retain the original Action2B dimensions")
    cfg.model._target_ = "vpp2.meanflow.runtime.create_model"
    cfg.model.benchmark = cfg.benchmark
    cfg.model.meanflow = "${meanflow}"
    fresh = cfg.meanflow.student_init.format == "baseline_init"
    cfg.model.action_dit_pretrained_path = cfg.meanflow.student_init.action_checkpoint if fresh else None
    cfg.model.skip_dit_load_from_pretrain = not fresh
    cfg.meanflow.budget_steps = int(cfg.max_steps)
    cfg.eval_num_inference_steps = 1
    cfg.evaluation.num_inference_steps = 1
    # For LIBERO export the exact paired, frozen Video source.
    if cfg.benchmark == "libero":
        cfg.frozen_video_checkpoint = cfg.meanflow.student_init.get(
            "video_checkpoint", cfg.model.dit_checkpoint_path
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--benchmark", required=True, choices=["robodojo", "libero"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-batch-change", action="store_true")
    args, overrides = parser.parse_known_args()
    if any("=" not in x for x in overrides):
        parser.error("Remaining arguments must be key=value overrides")
    cfg = load_config(args.config, overrides)
    if args.benchmark != cfg.benchmark:
        raise ValueError("Config and requested benchmark differ")
    world = int(os.environ.get("WORLD_SIZE", 1))
    batch = world * cfg.batch_size * cfg.gradient_accumulation_steps
    summary = dict(
        benchmark=cfg.benchmark,
        objective=cfg.meanflow.objective,
        world_size=world,
        global_batch=batch,
        micro_batch=int(cfg.batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        num_workers=int(cfg.num_workers),
        learning_rate=float(cfg.learning_rate),
        max_steps=int(cfg.max_steps),
        save_every=int(cfg.save_every),
        action_sampler=cfg.inference.action_sampler,
        reference_global_batch=cfg.reference_global_batch_size,
        train_mode=cfg.train_mode,
        action_nfe=1,
        derivative_backend=cfg.meanflow.derivative_backend,
        student_init=OmegaConf.to_container(cfg.meanflow.student_init, resolve=True),
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if batch != cfg.reference_global_batch_size and not args.allow_batch_change:
        raise ValueError(
            "Global batch changed; use --allow-batch-change only for a named diagnostic"
        )
    paths = [cfg.model.model_id, cfg.data.train.pretrained_norm_stats]
    paths.extend(
        [cfg.data.train.metadata_path]
        if cfg.benchmark == "robodojo"
        else list(cfg.data.train.dataset_dirs)
    )

    def init_paths(spec):
        return (
            [spec.full_checkpoint]
            if spec.format == "full_training"
            else [spec.action_checkpoint, spec.video_checkpoint]
        )

    paths.extend(init_paths(cfg.meanflow.student_init))
    if cfg.meanflow.objective == "teacher_trajectory":
        paths.extend(init_paths(cfg.meanflow.teacher))
    missing = [str(p) for p in paths if not p or not Path(str(p)).exists()]
    if missing:
        raise FileNotFoundError("Required assets missing:\n" + "\n".join(missing))
    out = Path(cfg.output_dir)
    if (out / "config.yaml").exists() and not cfg.resume and int(os.environ.get("RANK", 0)) == 0:
        raise FileExistsError("Choose a new output_dir or resume the existing state")
    if cfg.benchmark == "robodojo":
        from vpp2.runtime import run_training
    else:
        from vpp2.libero.runtime import run_training
    run_training(cfg)


if __name__ == "__main__":
    main()
