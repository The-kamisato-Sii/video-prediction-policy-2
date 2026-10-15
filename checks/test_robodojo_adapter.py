"""Adapter startup tests; real-model transport/physics are separate probes."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("defer", [False, True])
def test_startup_loads_once_and_preserves_first_observation(monkeypatch, defer):
    base = types.ModuleType("XPolicyLab.model_template")
    base.ModelTemplate = type("ModelTemplate", (), {})
    monkeypatch.setitem(sys.modules, "XPolicyLab", types.ModuleType("XPolicyLab"))
    monkeypatch.setitem(sys.modules, "XPolicyLab.model_template", base)
    spec = importlib.util.spec_from_file_location(
        "vpp2_test_adapter", Path(__file__).parents[1] / "policy/VPP2/model.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    import vpp2.policy

    observations, loads, releases = [], [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: releases.append(True))
    policy = types.SimpleNamespace(
        action_horizon=32, replan_steps=24, history_condition_frames=0,
        observe=observations.append, reset=lambda: None,
        _infer_action_chunk=lambda obs, instruction: np.zeros((32, 16), np.float32),
    )

    def load(cfg):
        loads.append(cfg)
        return policy

    monkeypatch.setattr(vpp2.policy, "get_model", load)
    model = adapter.Model({
        "action_type": "ee", "env_cfg_type": "arx_x5",
        "checkpoint_path": "unused/action.pt", "dataset_stats_path": "unused/stats.json",
        "defer_model_load": defer,
        "release_cuda_cache": True,
    })
    model.reset()
    assert len(loads) == int(not defer)
    assert len(releases) == int(not defer)
    assert model.action_horizon == 32
    obs = {
        "env_idx": 0, "instruction": "stack bowls",
        "vision": {k: {"color": np.zeros((240, 320, 3), np.uint8)} for k in (
            "cam_head", "cam_left_wrist", "cam_right_wrist"
        )},
        "state": {f"{side}_{key}": np.array(value, np.float32)
                  for side in ("left", "right") for key, value in (
                      ("ee_pose", [0, 0, 0.3, 1, 0, 0, 0]), ("ee_joint_state", [0.5])
                  )},
    }
    model.update_obs(obs)
    assert len(loads) == 1
    assert len(observations) == 1
    assert observations[0]["joint_action"]["vector"].shape == (16,)
    assert len(model.get_action()) == 24
    assert len(releases) == 2
    model.update_obs(obs)
    assert len(loads) == 1
    assert len(observations) == 2
