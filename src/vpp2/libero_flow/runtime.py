"""Read baseline configuration without changing its factory or launcher."""
from pathlib import Path
import os
import json
from omegaconf import OmegaConf


def load_config(overrides=()):
    base = OmegaConf.load("configs/libero/train_action.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.load("scripts/libero_flow/train.yaml"), OmegaConf.from_dotlist(list(overrides)))
    for key in ("model_id", "dit_checkpoint_path", "tokenizer_model_id", "action_dit_pretrained_path"):
        if cfg.model.get(key):
            cfg.model[key] = str(Path(cfg.model[key]).resolve())
    return cfg


def create_model(cfg, device="cuda"):
    import torch
    # Check the codec's optional image dependencies before loading the14B model.
    from diffusers import AutoencoderKL  # noqa: F401
    from vpp2.libero.runtime import create_action_model
    from .model import LiberoFlowActionModel
    from .codec import MotionCodec
    objective = cfg.flow.get("objective", "latent_flow_matching")
    model_class, codec_class = LiberoFlowActionModel, MotionCodec
    if objective == "direct_reconstruction":
        from .reconstruction import LiberoFlowReconstructionModel
        from .reconstruction_codec import ReconstructionCodec
        model_class, codec_class = LiberoFlowReconstructionModel, ReconstructionCodec
    elif objective != "latent_flow_matching":
        raise ValueError(f"Unknown flow objective: {objective}")
    kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    kwargs.pop("_target_")
    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    model = create_action_model(**kwargs, model_dtype=dtype, device=device, _model_class=model_class)
    codec = codec_class(cfg.flow.codec_path, frame_batch=cfg.flow.codec_frame_batch,
                        cameras=cfg.flow.get("motion_cameras", ["image", "wrist_image"]))
    extra = dict(reconstruction_loss=cfg.flow.get("reconstruction_loss", "l1"),
                 scene_flow_loss_weight=cfg.flow.get("scene_flow_loss_weight", 1.0)) if objective == "direct_reconstruction" else {}
    return model.configure_flow(codec, cfg.flow.tokens, cfg.flow.decoded_3d_weight, **extra)


def create_dataset(cfg):
    from .data import FlowActionDataset
    source = cfg.flow.get("motion_source", "simulator_gt")
    if source == "4rc_v1":
        from vpp2.libero_pseudo_flow.training import PseudoFlowActionDataset
        FlowActionDataset = PseudoFlowActionDataset
    elif source != "simulator_gt":
        raise ValueError(f"Unknown motion source: {source}")
    if cfg.flow.motion_normalization != "selected_views_clip_mean_radius_v1":
        raise ValueError("Expected selected-view clip normalization for flow labels")
    return FlowActionDataset(cfg.data.train, denominator=int(cfg.flow.action_denominator),
        flow_dt=int(cfg.flow.dt), action_only=bool(cfg.flow.action_only),
        missing_motion=str(cfg.flow.missing_motion), available_manifest=cfg.flow.get("available_manifest"),
        motion_backend=str(cfg.flow.get("motion_backend", "auto")),
        motion_open_files=int(cfg.flow.get("motion_open_files", 8)),
        flow_episode_scope=str(cfg.flow.get("flow_episode_scope", "all")),
        cameras=cfg.flow.get("motion_cameras", ["image", "wrist_image"]))


def train(cfg):
    import torch
    from accelerate.utils import set_seed
    from vpp2.utils import misc
    from vpp2.utils.logging_config import setup_logging
    from vpp2.utils.training_profiler import TrainingProfiler
    from .trainer import LiberoFlowTrainer
    from .diagnostics import FlowActionDiagnosticEvaluator
    out = Path(cfg.output_dir)
    previous_config = out / "resolved_config.yaml"
    if cfg.resume and previous_config.is_file():
        previous = OmegaConf.load(previous_config)
        if previous.flow.get("motion_source", "simulator_gt") != cfg.flow.get("motion_source", "simulator_gt"):
            raise ValueError("Cannot resume across motion label sources; start a new run")
        if previous.flow.get("motion_normalization", "legacy_dual_view") != cfg.flow.motion_normalization:
            raise ValueError("Cannot resume across motion normalization changes; start a new run")
        if list(previous.flow.get("motion_cameras", ["image", "wrist_image"])) != list(cfg.flow.get("motion_cameras", ["image", "wrist_image"])):
            raise ValueError("Cannot resume across motion camera selections; start a new run")
        old_scope = previous.flow.get("flow_episode_scope", "complementary")
        new_scope = cfg.flow.get("flow_episode_scope", "all")
        if old_scope != new_scope:
            raise ValueError("Cannot resume sampler/optimizer state across flow episode scopes; start a new run")
    global_rank = int(os.environ.get("RANK", 0))
    # Only rank0 owns output validation/writes. Another rank may arrive after
    # rank0 has written resolved_config.yaml during this same launch.
    if global_rank == 0 and out.exists() and any(out.iterdir()) and not cfg.resume and not cfg.overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    setup_logging()
    misc.register_work_dir(out)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    set_seed(int(cfg.seed))
    if global_rank == 0:
        OmegaConf.save(cfg, out / "resolved_config.yaml")
    profiler = TrainingProfiler.from_config(cfg)
    with profiler.measure_startup("dataset_build_ms"):
        dataset = create_dataset(cfg)
    if global_rank == 0:
        (out / "dataset_audit.json").write_text(json.dumps(dataset.audit, indent=2)+'\n')
    with profiler.measure_startup("model_build_ms", cuda=True):
        model = create_model(cfg, device=f"cuda:{rank}")
    evaluator = FlowActionDiagnosticEvaluator(dataset, out / "eval", seed=cfg.seed) if int(cfg.eval_every)>0 else None
    trainer = LiberoFlowTrainer(model, dataset, cfg=cfg, evaluator=evaluator, profiler=profiler)
    trainer.train()
    memory = torch.tensor([torch.cuda.max_memory_allocated()/2**30], device=trainer.accelerator.device)
    memories = trainer.accelerator.gather(memory).cpu().tolist()
    if trainer.accelerator.is_main_process:
        report = dict(status="passed", optimizer_steps=trainer.global_step,
            world_size=trainer.accelerator.num_processes, tokens=int(cfg.flow.tokens),
            global_batch=int(cfg.batch_size)*int(cfg.gradient_accumulation_steps)*trainer.accelerator.num_processes,
            micro_batch=int(cfg.batch_size), gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
            peak_allocated_gib=memories, shapes=model.last_shapes)
        report["flow_objective"] = cfg.flow.get("objective", "latent_flow_matching")
        (out / "completion.json").write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)
    cfg = load_config(args.overrides)
    world = int(os.environ.get("WORLD_SIZE", 1))
    actual = int(cfg.batch_size)*int(cfg.gradient_accumulation_steps)*world
    if actual != 256 and not args.diagnostic:
        raise ValueError(f"Global batch must be256, got{actual}")
    if int(cfg.max_steps) != 30000 and not args.diagnostic:
        raise ValueError("Only labelled diagnostics may change30000 optimizer steps")
    if cfg.flow.missing_motion not in ("error", "skip"):
        raise ValueError("flow.missing_motion must be error or skip")
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    train(cfg)


if __name__ == "__main__":
    main()
