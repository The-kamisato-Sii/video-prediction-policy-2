"""Strict FM initialization, paired checkpoints, provenance and training construction."""

import hashlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .model import MeanFlowLibero, MeanFlowVPP2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fm(model, spec, benchmark):
    expected_step = 100000 if benchmark == "robodojo" else 30000
    fmt = spec["format"]
    if fmt == "full_training":
        path = Path(spec["full_checkpoint"]).resolve()
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if payload.get("meanflow") or "mot" not in payload:
            raise ValueError("Expected an original full FM training checkpoint")
        if payload.get("step") != expected_step:
            raise ValueError(f"Main {benchmark} initialization requires FM step {expected_step}")
        model.mot.load_state_dict(payload["mot"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return dict(
            format=fmt, full_checkpoint=str(path), sha256=sha256(path), source_step=payload["step"]
        )
    if fmt != "deployment_pair":
        raise ValueError("Initialization format must be full_training or deployment_pair")
    ap, vp = Path(spec["action_checkpoint"]).resolve(), Path(spec["video_checkpoint"]).resolve()
    action = torch.load(ap, map_location="cpu", mmap=True, weights_only=True)
    video = torch.load(vp, map_location="cpu", mmap=True, weights_only=True)
    from vpp2.checkpoint_compat import action_config

    action_config(action)
    if action.get("meanflow"):
        raise ValueError("Main experiments must start from an FM checkpoint, not an MF warm start")
    if benchmark == "robodojo" and action["step"] != video["step"]:
        raise ValueError("RoboDojo Video/Action must be a matched joint pair")
    if (
        benchmark == "libero"
        and action.get("video_checkpoint", {}).get("size_bytes", vp.stat().st_size)
        != vp.stat().st_size
    ):
        raise ValueError("LIBERO frozen Video does not match checkpoint reference size")
    if action.get("step") != expected_step:
        raise ValueError(f"Main {benchmark} initialization requires FM step {expected_step}")
    if benchmark == "libero" and video.get("step") != 10000:
        raise ValueError("LIBERO requires the paired frozen Video-10k")
    model._load_video_base_state_dict(video["dit"])
    model.action_expert.load_state_dict(action["action_expert"], strict=True)
    model.proprio_encoder.load_state_dict(action["proprio_encoder"], strict=True)
    return dict(
        format=fmt,
        action_checkpoint=str(ap),
        video_checkpoint=str(vp),
        action_sha256=sha256(ap),
        video_sha256=sha256(vp),
        source_step=action["step"],
        video_source_step=video["step"],
    )


def create_model(benchmark, meanflow, **kwargs):
    from vpp2.libero.runtime import create_action_model
    from vpp2.runtime import create_vpp2_wan21_14b

    cfg = (
        OmegaConf.to_container(meanflow, resolve=True)
        if isinstance(meanflow, DictConfig)
        else dict(meanflow)
    )
    factory = create_vpp2_wan21_14b if benchmark == "robodojo" else create_action_model
    cls = MeanFlowVPP2 if benchmark == "robodojo" else MeanFlowLibero
    spec = cfg["student_init"]
    if spec["format"] == "baseline_init":
        if benchmark != "libero" or cfg["objective"] not in {"data_meanflow", "data_improved_meanflow", "direct_action"}:
            raise ValueError("baseline_init is for LIBERO one-step controlled experiments")
        kwargs.update(action_dit_pretrained_path=str(Path(spec["action_checkpoint"]).resolve()),
                      dit_checkpoint_path=str(Path(spec["video_checkpoint"]).resolve()),
                      skip_dit_load_from_pretrain=False)
        action_init = torch.load(kwargs["action_dit_pretrained_path"], map_location="cpu", mmap=True, weights_only=True)
        video_init = torch.load(kwargs["dit_checkpoint_path"], map_location="cpu", mmap=True, weights_only=True)
        if action_init.get("format") != "wan21_action_v1" or action_init.get("policy", {}).get("source_step") != 10000:
            raise ValueError("Controlled LIBERO runs require fresh Action initialization from Video10k")
        if video_init.get("step") != 10000 or "dit" not in video_init:
            raise ValueError("Controlled LIBERO runs require standalone Video10k")
        del action_init, video_init
        student = factory(**kwargs, _model_class=cls)
        source = dict(format="baseline_init", source_step=0, video_source_step=10000,
                      action_checkpoint=kwargs["action_dit_pretrained_path"],
                      video_checkpoint=kwargs["dit_checkpoint_path"])
        student.configure_meanflow(cfg, dict(student=source, teacher=None))
        return student
    # Never pass a release action.pt to the fresh backbone initializer.
    kwargs.update(action_dit_pretrained_path=None, skip_dit_load_from_pretrain=True)
    student = factory(**kwargs, _model_class=cls)
    source = load_fm(student, spec, benchmark)
    teacher = None
    teacher_source = None
    if cfg["objective"] == "teacher_trajectory":
        teacher_spec = dict(cfg["teacher"])
        teacher_spec.setdefault("format", "deployment_pair")
        teacher = factory(**kwargs)
        teacher_source = load_fm(teacher, teacher_spec, benchmark)
        for key in ("action_sha256", "video_sha256", "sha256"):
            if source.get(key) != teacher_source.get(key):
                raise ValueError(
                    "Teacher and student must initialize from the same paired FM weights"
                )
        teacher.eval().requires_grad_(False)
    student.configure_meanflow(cfg, dict(student=source, teacher=teacher_source), teacher)
    return student
