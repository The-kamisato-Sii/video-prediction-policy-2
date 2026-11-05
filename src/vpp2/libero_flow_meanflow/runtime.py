"""Read baseline configuration without changing its factory or launcher."""
from pathlib import Path
import os
import json
from omegaconf import OmegaConf


VARIANTS = {'adaptive': 'train_meanflow.yaml', 'diagonal': 'train_meanflow_diagonal_fix.yaml',
            'imf_boundary': 'train_imf_boundary.yaml', 'imf_auxiliary': 'train_imf_auxiliary.yaml'}

def _inherited(path):
    c = OmegaConf.load(path)
    base = c.pop('_base_', None)
    return OmegaConf.merge(_inherited(path.parent/base), c) if base else c

def load_config(overrides=()):
    from vpp2.libero_flow.runtime import load_config as base_config
    cfg = base_config(overrides)
    variant = cfg.get('flow_meanflow_variant', 'adaptive')
    if variant not in VARIANTS: raise ValueError(variant)
    original = _inherited(Path('configs/libero')/VARIANTS[variant])
    cfg = OmegaConf.merge(cfg, {'meanflow': original.meanflow, 'inference': original.inference},
                          OmegaConf.from_dotlist(list(overrides)))
    cfg.flow.motion_source = cfg.flow.get('motion_source', 'simulator_gt')
    if cfg.flow.tokens != 32 or cfg.flow.action_denominator != 8:
        raise ValueError('This controlled matrix is 32 tokens, action1/8')
    if cfg.flow.batch_sampling != 'homogeneous_interleaved':
        raise ValueError('Every rank must train the same branch in each optimizer step')
    if cfg.meanflow.derivative_backend != 'jvp': raise ValueError('Use exact baseline JVP')
    return cfg


def create_model(cfg, device="cuda"):
    import torch
    # Check the codec's optional image dependencies before loading the14B model.
    from diffusers import AutoencoderKL  # noqa: F401
    from vpp2.libero.runtime import create_action_model
    from .model import LiberoFlowMeanFlow
    from vpp2.libero_flow.codec import MotionCodec
    objective = cfg.flow.get("objective", "latent_meanflow")
    model_class, codec_class = LiberoFlowMeanFlow, MotionCodec
    if objective == "direct_reconstruction":
        from .model import LiberoReconMeanFlow
        from vpp2.libero_flow.reconstruction_codec import ReconstructionCodec
        model_class, codec_class = LiberoReconMeanFlow, ReconstructionCodec
    elif objective != "latent_meanflow":
        raise ValueError(f"Unknown flow objective: {objective}")
    kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    kwargs.pop("_target_")
    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    model = create_action_model(**kwargs, model_dtype=dtype, device=device, _model_class=model_class)
    codec = codec_class(cfg.flow.codec_path, frame_batch=cfg.flow.codec_frame_batch,
                        cameras=cfg.flow.get("motion_cameras", ["image", "wrist_image"]))
    extra = dict(reconstruction_loss=cfg.flow.get("reconstruction_loss", "l1"),
                 scene_flow_loss_weight=cfg.flow.get("scene_flow_loss_weight", 1.0)) if objective == "direct_reconstruction" else {}
    model.configure_flow(codec, cfg.flow.tokens, cfg.flow.decoded_3d_weight, **extra)
    return model.configure_meanflow(cfg.meanflow)


def create_dataset(cfg):
    from vpp2.libero_flow.runtime import create_dataset as original
    return original(cfg)


def train(cfg):
    import torch
    from accelerate.utils import set_seed
    from vpp2.utils import misc
    from vpp2.utils.logging_config import setup_logging
    from vpp2.utils.training_profiler import TrainingProfiler
    from vpp2.libero_flow.trainer import LiberoFlowTrainer
    from .diagnostics import FlowActionDiagnosticEvaluator
    out = Path(cfg.output_dir)
    previous_config = out / "resolved_config.yaml"
    if cfg.resume and previous_config.is_file():
        previous = OmegaConf.load(previous_config)
        if OmegaConf.to_container(previous.meanflow, resolve=True) != OmegaConf.to_container(cfg.meanflow, resolve=True):
            raise ValueError('Cannot resume across MF/iMF changes')
        if previous.flow != cfg.flow:
            raise ValueError('Cannot resume across flow configuration changes')
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
        report["flow_objective"] = cfg.flow.get("objective", "latent_meanflow")
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
