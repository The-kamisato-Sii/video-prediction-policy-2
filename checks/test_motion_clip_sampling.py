"""Keep flow intervals independent of source sampling and action-window bounds."""
import h5py
import numpy as np
import pytest

from vpp2.libero.motion_data import load_motion_clip
from vpp2.libero.motion_replay import REPLAY_VERSION


@pytest.fixture
def motion_file(tmp_path):
    path = tmp_path / "motion.hdf5"
    frames = 33
    t = np.arange(frames)
    yy, xx = np.mgrid[:2, :3]
    with h5py.File(path, "w") as f:
        f.attrs["replay_version"] = REPLAY_VERSION
        f["frame_index"] = t
        for view, key in enumerate(("image", "wrist_image")):
            group = f.create_group(key)
            points = np.zeros((frames, 2, 3, 3), np.float32)
            points[..., 0] = t[:, None, None] ** 2 + xx
            points[..., 1] = yy + view
            points[..., 2] = 2
            group["point_map"] = points
            group["valid_mask"] = np.ones((frames, 2, 3), bool)
            for dt in (1, 4):
                flow_group = group.create_group(f"flow_dt{dt}")
                flow = np.zeros_like(points)
                source = t[:-dt]
                flow[:-dt, ..., 0] = ((source + dt) ** 2 - source ** 2)[:, None, None]
                valid = np.ones((frames, 2, 3), bool)
                valid[-dt:] = False
                flow_group["scene_flow"] = flow
                flow_group["valid_mask"] = valid
    return path


@pytest.mark.parametrize("dt,last_target,last_displacement", [(1, 29, 57), (4, 32, 240)])
def test_eight_sources_keep_eighth_flow(motion_file, dt, last_target, last_displacement):
    clip = load_motion_clip(motion_file, length=8, stride=4, flow_dt=dt,
                            prediction_horizon=32, normalize="none")
    np.testing.assert_array_equal(clip["requested_frame_index"], np.arange(0, 32, 4))
    np.testing.assert_array_equal(clip["flow_target_frame_index"], np.arange(0, 32, 4) + dt)
    assert clip["flow_target_frame_index"][-1] == last_target
    for view in clip["cameras"].values():
        assert view["point_map"].shape[0] == view["scene_flow"].shape[0] == 8
        assert view["valid_mask"].all()
        assert view["flow_valid_mask"].all()
        np.testing.assert_array_equal(view["scene_flow"][-1, ..., 0], last_displacement)


def test_interval_change_preserves_geometry_and_normalization(motion_file):
    clips = [load_motion_clip(motion_file, length=8, stride=4, flow_dt=dt,
                             prediction_horizon=32) for dt in (1, 4)]
    np.testing.assert_array_equal(clips[0]["center"], clips[1]["center"])
    assert clips[0]["scale"] == clips[1]["scale"]
    for key in ("image", "wrist_image"):
        np.testing.assert_array_equal(clips[0]["cameras"][key]["point_map"],
                                      clips[1]["cameras"][key]["point_map"])
        assert not np.array_equal(clips[0]["cameras"][key]["scene_flow"],
                                  clips[1]["cameras"][key]["scene_flow"])


def test_endpoint_bound_does_not_remove_source_geometry(motion_file):
    for dt, last_valid in ((1, True), (4, False)):
        clip = load_motion_clip(motion_file, length=8, stride=4, flow_dt=dt,
                                prediction_horizon=29, normalize="none")
        for view in clip["cameras"].values():
            assert view["valid_mask"][-1].all()
            assert bool(view["flow_valid_mask"][-1].all()) == last_valid
            if not last_valid:
                np.testing.assert_array_equal(view["scene_flow"][-1], 0)


def test_episode_end_masks_missing_endpoint_even_inside_horizon(motion_file):
    clip = load_motion_clip(motion_file, start=4, length=8, stride=4, flow_dt=4,
                            prediction_horizon=32, normalize="none")
    for view in clip["cameras"].values():
        assert view["valid_mask"].all()
        assert view["flow_valid_mask"][:-1].all()
        assert not view["flow_valid_mask"][-1].any()
        np.testing.assert_array_equal(view["scene_flow"][-1], 0)


def test_legacy_call_keeps_terminal_flow_mask(motion_file):
    clip = load_motion_clip(motion_file, length=9, stride=4, normalize="none")
    assert clip["flow_dt"] == 4
    assert clip["prediction_horizon"] == 32
    for view in clip["cameras"].values():
        assert view["flow_valid_mask"][:-1].all()
        assert not view["flow_valid_mask"][-1].any()


def test_reject_source_past_prediction_window(motion_file):
    with pytest.raises(ValueError, match="last source frame"):
        load_motion_clip(motion_file, length=8, stride=4, prediction_horizon=27)
