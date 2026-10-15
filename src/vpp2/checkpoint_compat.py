"""Read historical checkpoints while emitting only VPP2 configuration names.

Legacy identifiers below describe serialized input formats, not current methods.
No legacy package is imported and tensor keys are unchanged.
"""

ACTION_FORMAT = "vpp2_wan21_action_v1"
_LEGACY_ACTION_FORMATS = {"fastwam_wan21_action_v1"}
_LEGACY_NAMESPACES = ("fastwam.", "robodojo_joint2b.")
_LEGACY_TARGET_PARTS = {
    "fastwam": "vpp2",
    "FastWAM": "VPP2",
    "fastwam_processor": "vpp2_processor",
    "FastWAMProcessor": "VPP2Processor",
    "create_fastwam_wan21_14b": "create_vpp2_wan21_14b",
}


def canonicalize_config(value):
    """Convert Hydra targets recursively, preserving paths and other values."""
    if isinstance(value, list):
        return [canonicalize_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonicalize_config(item) for key, item in value.items()}
    target = result.get("_target_")
    if isinstance(target, str):
        for prefix in _LEGACY_NAMESPACES:
            if target.startswith(prefix):
                target = "vpp2." + target[len(prefix) :]
                break
        if target.startswith("vpp2."):
            target = ".".join(_LEGACY_TARGET_PARTS.get(part, part) for part in target.split("."))
        result["_target_"] = target
    return result


def action_config(payload):
    if payload.get("format") not in {ACTION_FORMAT, *_LEGACY_ACTION_FORMATS}:
        raise ValueError("Expected an exported VPP2 action checkpoint")
    return canonicalize_config(payload["resolved_config"])


def configure_fm_continuation(model, cfg):
    """Opt-in B2 provenance, retained across full-state resumes and compact exports."""
    if not cfg.get("fm_continuation", False):
        return
    import json
    import os
    from pathlib import Path
    import torch
    from .meanflow.runtime import sha256

    if cfg.get("meanflow"):
        raise ValueError("FM continuation must use the original FM objective")
    manifest = Path(cfg.output_dir) / "fm_continuation.json"
    if cfg.resume:
        source = json.loads(manifest.read_text())
    else:
        if not cfg.resume_ckpt:
            raise ValueError("FM continuation requires resume_ckpt as its parameter initializer")
        checkpoint = Path(cfg.resume_ckpt).resolve()
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        expected = 30000 if cfg.train_mode == "action_only" else 100000
        if payload.get("step") != expected or payload.get("meanflow"):
            raise ValueError(f"B2 requires the original FM step {expected}")
        source = dict(
            source_step=expected,
            source_checkpoint=str(checkpoint),
            source_sha256=sha256(checkpoint),
            action_sampler="flow_matching",
        )
        if cfg.train_mode == "action_only":
            video = Path(cfg.frozen_video_checkpoint).resolve()
            source["video_sha256"] = sha256(video)
        if int(os.environ.get("RANK", 0)) == 0:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            temp = manifest.with_suffix(".tmp")
            temp.write_text(json.dumps(source, indent=2) + "\n")
            temp.replace(manifest)
    model.fm_continuation_source = source
