from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from vpp2.libero import cli, runtime


def test_only_writer_checks_existing_output(tmp_path, monkeypatch):
    cfg = OmegaConf.load("configs/libero/train_action.yaml")
    cfg.output_dir = str(tmp_path / "output")
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.yaml").write_text("rank_zero_has_written_this_launch: true\n")
    cfg.model.model_id = str(tmp_path)
    cfg.model.dit_checkpoint_path = str(tmp_path / "video.pt")
    cfg.model.action_dit_pretrained_path = str(tmp_path / "action.pt")
    cfg.data.train.pretrained_norm_stats = str(tmp_path / "stats.json")
    cfg.data.train.dataset_dirs = [str(tmp_path)] * 4
    cfg.data.train.text_embedding_cache_dir = str(tmp_path)
    for name in ("video.pt", "action.pt", "stats.json"):
        (tmp_path / name).touch()
    config = tmp_path / "input.yaml"
    OmegaConf.save(cfg, config)
    args = SimpleNamespace(config=config, overrides=[], dry_run=False, allow_batch_change=False)
    calls = []
    monkeypatch.setattr(runtime, "run_training", lambda config: calls.append(config))
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "1")
    cli.train(args)
    assert len(calls) == 1
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(FileExistsError, match="Output contains an existing run"):
        cli.train(args)
    assert len(calls) == 1
