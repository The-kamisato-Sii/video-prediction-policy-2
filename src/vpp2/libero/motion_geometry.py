"""Dense rigid-surface scene flow in a fixed reference camera frame.

MotionCrafter convention: X_i is indexed by source pixels, and V_i transports
that same material point to the next time. Camera ego-motion is excluded.
"""
import numpy as np


def transform_points(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def invert_pose(pose):
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def unproject(depth, intrinsic, camera_to_world):
    """Pixel-center pinhole backprojection; metric depth is optical-axis Z."""
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(intrinsic).T
    return transform_points(rays * depth[..., None], camera_to_world)


def advect_rigid(points_world, body_ids, rotations, positions, next_rotations, next_positions):
    """Follow each visible surface sample using its own articulated body."""
    ids = np.maximum(body_ids, 0)
    rotation = rotations[ids]
    # Row-vector implementation of R_i^T (X - t_i).
    local = np.einsum("...ji,...j->...i", rotation, points_world - positions[ids])
    return np.einsum("...ij,...j->...i", next_rotations[ids], local) + next_positions[ids]


def scene_flow(points_world, body_ids, current, target, reference_to_world, valid):
    moved = advect_rigid(points_world, body_ids, *current, *target)
    world_to_reference = invert_pose(reference_to_world)
    points = transform_points(points_world, world_to_reference)
    vectors = (moved - points_world) @ world_to_reference[:3, :3].T
    points[~valid] = 0
    vectors[~valid] = 0
    return points.astype(np.float32), vectors.astype(np.float32), moved


def target_visibility(points_world, depth, intrinsic, camera_to_world, valid,
                      target_body_ids=None, source_body_ids=None, tolerance=0.005):
    """Additional endpoint visibility mask; occlusion does not invalidate 3D flow."""
    camera_points = transform_points(points_world, invert_pose(camera_to_world))
    pixels = camera_points @ intrinsic.T
    z = camera_points[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = pixels[..., :2] / pixels[..., 2:]
    uv = np.nan_to_num(uv, nan=-1e6, posinf=-1e6, neginf=-1e6)
    u, v = np.rint(uv).astype(np.int64).transpose(2, 0, 1)
    h, w = depth.shape
    inside = valid & (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uu, vv = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    visible = inside & (np.abs(depth[vv, uu] - z) <= tolerance)
    if target_body_ids is not None:
        visible &= target_body_ids[vv, uu] == source_body_ids
    return visible


def canonical_normalization(point_maps, valid_masks, epsilon=1e-8):
    """One center and isotropic mean-radius scale for the entire sequence."""
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for points, valid in zip(point_maps, valid_masks):
        xyz = np.asarray(points)[np.asarray(valid, dtype=bool)]
        sums += xyz.sum(axis=0, dtype=np.float64)
        count += len(xyz)
    if count == 0:
        raise ValueError("Cannot normalize an empty point sequence")
    center = sums / count
    radius_sum = 0.0
    for points, valid in zip(point_maps, valid_masks):
        xyz = np.asarray(points)[np.asarray(valid, dtype=bool)]
        radius_sum += np.linalg.norm(xyz - center, axis=-1).sum(dtype=np.float64)
    return center, radius_sum / count + epsilon


def normalize_geometry(points, flow, center, scale, valid):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be finite and positive")
    x = (points - center) / scale
    v = flow / scale
    x[~valid] = 0
    v[~valid] = 0
    return x.astype(np.float32), v.astype(np.float32)
