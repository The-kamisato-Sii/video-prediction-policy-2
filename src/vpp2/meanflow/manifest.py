"""Persist provenance without claiming that a short probe is a benchmark result."""

import json
import os
import platform
from pathlib import Path

import torch
from omegaconf import OmegaConf


def write_manifest(cfg, model):
    if int(os.environ.get("RANK", 0)) != 0 or not hasattr(model, "meanflow_metadata"):
        return
    from .runtime import sha256

    data = cfg.data.train
    files = {}
    for key in ("metadata_path", "pretrained_norm_stats"):
        value = data.get(key)
        if isinstance(value, str) and Path(value).is_file():
            files[key] = dict(path=str(Path(value).resolve()), sha256=sha256(value))
    manifest = dict(
        meanflow=model.meanflow_metadata(0),
        seed=int(cfg.seed),
        dataset_files=files,
        data_config=OmegaConf.to_container(data, resolve=True),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        global_batch=int(cfg.batch_size)
        * int(cfg.gradient_accumulation_steps)
        * int(os.environ.get("WORLD_SIZE", 1)),
        mixed_precision=cfg.mixed_precision,
        derivative_attention="SDPA math; primary retains baseline SDPA",
        geometry=False,
        benchmark_results=None,
    )
    destination = Path(cfg.output_dir) / "manifest.json"
    if destination.exists() and not cfg.resume:
        raise FileExistsError(destination)
    if not destination.exists():
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["meanflow"], indent=2), flush=True)
