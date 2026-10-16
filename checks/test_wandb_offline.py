"""Exercise the trainer's real offline writer without allocating any GPU."""
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

wandb = pytest.importorskip("wandb")

from vpp2.trainer import JointTrainer


def test_offline_trainer_records_config_metrics_and_step(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    trainer = JointTrainer.__new__(JointTrainer)
    trainer.cfg = OmegaConf.load("configs/libero/train_action.yaml")
    trainer.cfg.wandb.enabled = True
    trainer.cfg.wandb.mode = "offline"
    trainer.cfg.wandb.project = "vpp2-offline-check"
    trainer.cfg.wandb.name = "cpu-smoke"
    trainer.cfg.batch_size = 16
    trainer.wandb_enabled = True
    trainer.wandb_run = None
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.output_dir = str(tmp_path)
    trainer.global_step = 10
    trainer._init_wandb()
    assert trainer.wandb_run is None

    trainer.accelerator.is_main_process = True
    trainer._init_wandb()
    try:
        assert trainer.wandb_run.offline
        assert trainer.wandb_run.config["batch_size"] == 16
        assert trainer.wandb_run.config["save_every"] == 5000
        trainer._wandb_log({"train/loss": 0.25, "train/lr": 1e-4})
        trainer.wandb_run.log({}, commit=True)
        assert trainer.wandb_run.summary["train/loss"] == 0.25
        assert trainer.wandb_run.summary["_step"] == 10
    finally:
        trainer._finish_wandb()

    files = list(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0
