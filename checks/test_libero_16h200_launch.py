"""Validate the actual shell launcher's resolved recipe without GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize("rank", ["0", "1"])
def test_two_node_recipe_preserves_official_training(tmp_path, rank):
    capture = tmp_path / "capture-python"
    capture.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ('NODE_RANK', 'WORLD_SIZE', 'WANDB_MODE', 'MASTER_ADDR', 'MASTER_PORT')\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}))\n"
    )
    capture.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(capture), RANK=rank,
               MASTER_ADDR="10.0.0.1", MASTER_PORT="29616")
    env.pop("NODE_RANK", None)
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
    assert cfg.batch_size * cfg.gradient_accumulation_steps * 16 == 256
    assert cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 1
    assert cfg.num_workers == 16 and cfg.save_every == 5000
    assert cfg.wandb.enabled and cfg.wandb.mode == "offline"
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
