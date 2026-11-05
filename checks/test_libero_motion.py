"""Geometry invariants that catch camera-motion and correspondence mistakes."""
import numpy as np
import pytest

from vpp2.libero.motion_geometry import (advect_rigid, canonical_normalization,
    invert_pose, normalize_geometry, scene_flow, target_visibility,
    transform_points, unproject)
from vpp2.libero.motion_source import match_action_indices, retained_action_indices


def test_backprojection_and_pose_inverse():
    depth = np.full((3, 3), 2.0)
    k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    p = unproject(depth, k, pose)
    np.testing.assert_allclose(p[1, 1], [1, 2, 5])
    np.testing.assert_allclose(transform_points(p, invert_pose(pose))[0, 0], [-1, -1, 2])


def test_material_point_rotation_and_static_body():
    points = np.array([[[2., 0, 0], [0, 3, 0]]])
    ids = np.array([[1, 0]])
    rotations = np.tile(np.eye(3), (2, 1, 1))
    target_rotations = rotations.copy()
    target_rotations[1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    positions = np.zeros((2, 3))
    positions[1] = [1, 0, 0]
    target_positions = positions.copy()
    target_positions[1] = [3, 0, 0]
    moved = advect_rigid(points, ids, rotations, positions, target_rotations, target_positions)
    np.testing.assert_allclose(moved, [[[3, 1, 0], [0, 3, 0]]])


def test_static_scene_zero_flow_in_moving_camera_reference():
    points = np.array([[[0., 0, 2], [1, 1, 2]]])
    ids = np.zeros((1, 2), dtype=int)
    body = (np.eye(3)[None], np.zeros((1, 3)))
    ref = np.eye(4)
    ref[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ref[:3, 3] = [4, 5, 6]
    _, flow, _ = scene_flow(points, ids, body, body, ref, np.ones((1, 2), bool))
    np.testing.assert_array_equal(flow, 0)


def test_motion_normalization_uses_scale_without_center_or_clipping():
    p = np.array([[[0., 0, 0], [4, 0, 0], [99, 99, 99]]])
    valid = np.array([[True, True, False]])
    mu, scale = canonical_normalization([p], [valid], epsilon=0)
    np.testing.assert_allclose(mu, [2, 0, 0])
    assert scale == 2
    v = np.zeros_like(p)
    v[0, 0] = [10, 0, 0]
    pn, vn = normalize_geometry(p, v, mu, scale, valid)
    np.testing.assert_allclose(vn[0, 0], [5, 0, 0])
    np.testing.assert_array_equal(pn[~valid], 0)
    np.testing.assert_array_equal(vn[~valid], 0)
    np.testing.assert_allclose(pn[valid] * scale + mu, p[valid])


def test_occluded_surface_still_has_a_defined_motion():
    points = np.array([[[0., 0, 2], [4, 0, 2]]])
    k = np.eye(3)
    visible = target_visibility(points, np.ones((1, 2)), k, np.eye(4), np.ones((1, 2), bool))
    assert not visible.any()  # One point is behind foreground; the other leaves view.


def test_action_matching_preserves_removed_noop_indices():
    source = np.arange(35).reshape(5, 7).astype(float)
    indices = match_action_indices(source, source[[0, 1, 4]].astype(np.float32))
    np.testing.assert_array_equal(indices, [0, 1, 4])
    assert match_action_indices(source, source[::-1]) is None
    with pytest.raises(ValueError, match="Ambiguous"):
        match_action_indices(np.zeros((3, 7)), np.zeros((2, 7)))


def test_all_invalid_normalization_rejected():
    with pytest.raises(ValueError, match="empty"):
        canonical_normalization([np.zeros((2, 2, 3))], [np.zeros((2, 2), bool)])


def test_noop_rule_keeps_gripper_changes_and_uses_last_retained_control():
    actions = np.zeros((8, 7))
    actions[:, -1] = [-1, -1, -1, 1, 1, -1, -1, -1]
    actions[1, 0] = 0.1
    actions[6, 0] = 0.000099
    actions[7, 0] = 0.000101
    np.testing.assert_array_equal(retained_action_indices(actions), [1, 3, 5, 7])
