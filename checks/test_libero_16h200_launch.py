"""Validate the actual shell launcher's resolved recipe without GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize("rank", ["0", "1"])
@pytest.mark.parametrize("platform_prefix", ["", "PET_"])
def test_two_node_recipe_preserves_official_training(tmp_path, rank, platform_prefix):
    capture = tmp_path / "capture-python"
    capture.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ('NODE_RANK', 'WORLD_SIZE', 'WANDB_MODE', 'MASTER_ADDR', 'MASTER_PORT', 'VPP2_ACCELERATE_CONFIG')\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}))\n"
    )
    capture.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(capture), WANDB_PROJECT="test-libero",
               WANDB_NAME="test-action16")
    for key in ("NODE_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT", "NNODES", "NPROC_PER_NODE",
                "PET_NODE_RANK", "PET_MASTER_ADDR", "PET_MASTER_PORT", "PET_NNODES", "PET_NPROC_PER_NODE"):
        env.pop(key, None)
    env["PET_NODE_RANK" if platform_prefix else "RANK"] = rank
    env[platform_prefix + "MASTER_ADDR"] = "10.0.0.1"
    env[platform_prefix + "MASTER_PORT"] = "29616"
    env[platform_prefix + "NNODES"] = "2"
    env[platform_prefix + "NPROC_PER_NODE"] = "8"
    result = subprocess.run(
        ["bash", "scripts/libero/train_16h200.sh", "--dry-run"],
        env=env, text=True, capture_output=True, check=True,
    )
    captured = json.loads(result.stdout)
    args, launched_env = captured["args"], captured["env"]
    config_pos = args.index("--config")
    official = OmegaConf.load(args[config_pos + 1])
    overrides = [arg for arg in args[config_pos + 2:] if arg != "--dry-run"]
    cfg = OmegaConf.merge(official, OmegaConf.from_dotlist(overrides))
    assert launched_env["NODE_RANK"] == rank
    assert launched_env["WORLD_SIZE"] == "16"
    assert launched_env["WANDB_MODE"] == "offline"
    assert launched_env["MASTER_ADDR"] == "10.0.0.1"
    assert launched_env["MASTER_PORT"] == "29616"
    accelerate = OmegaConf.load(launched_env["VPP2_ACCELERATE_CONFIG"])
    ds = json.loads(Path(accelerate.deepspeed_config.deepspeed_config_file).read_text())
    assert ds["zero_optimization"]["stage"] == 1
    official_ds = json.loads(Path("configs/libero/deepspeed_zero2.json").read_text())
    ds["zero_optimization"]["stage"] = 2
    assert ds == official_ds
    assert cfg.batch_size * cfg.gradient_accumulation_steps * 16 == 256
    assert cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 1
    assert cfg.num_workers == 4 and cfg.save_every == 5000
    assert cfg.wandb.enabled and cfg.wandb.mode == "offline"
    assert cfg.wandb.project == "test-libero" and cfg.wandb.name == "test-action16"
    # Everything outside local paths, batch, output safety and tracking is identical.
    for key in ("output_dir", "overwrite", "batch_size", "wandb"):
        cfg[key] = official[key]
    for key in ("model_id", "dit_checkpoint_path", "action_dit_pretrained_path"):
        assert Path(cfg.model[key]).is_absolute()
        cfg.model[key] = official.model[key]
    assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(official, resolve=True)


@pytest.mark.parametrize("rank,addr", [("2", "10.0.0.1"), ("0", "127.0.0.1")])
def test_reject_invalid_node_layout(rank, addr):
    result = subprocess.run(
        ["bash", "scripts/libero/train_16h200.sh", "--dry-run"],
        env=dict(os.environ, NODE_RANK=rank, MASTER_ADDR=addr),
        text=True, capture_output=True,
    )
    assert result.returncode == 2
