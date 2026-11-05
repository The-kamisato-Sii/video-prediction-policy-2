"""Validate dense simulator motion and export a normalized 17-frame example."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from vpp2.libero.motion_data import load_motion_clip
from vpp2.libero.motion_geometry import transform_points, invert_pose

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--motion", required=True)
p.add_argument("--output", required=True)
p.add_argument("--clip-start", type=int, default=40)
p.add_argument("--normalized-clip", required=True)
a = p.parse_args()
report = {"cameras": {}, "issues": []}
with h5py.File(a.motion, "r") as f:
    rotations, positions = f["body_rotations"][:], f["body_positions"][:]
    ref = f["reference_camera_to_sim_world"][:]
    inv = invert_pose(ref)
    fixed = (np.max(np.abs(rotations - rotations[:1]), axis=(0, 2, 3)) < 1e-12)
    fixed &= np.max(np.abs(positions - positions[:1]), axis=(0, 2)) < 1e-12
    report["frames"] = len(f["frame_index"])
    report["metric_geometry_preserved"] = True
    for key in ("image", "wrist_image"):
        g = f[key]
        camera_report = dict(rgb_mae_mean=float(g["rgb_mae"][:].mean()), rgb_mae_max=float(g["rgb_mae"][:].max()), strides={})
        camera_poses = g["camera_to_reference"][:]
        camera_report["camera_translation_range_metres"] = float(np.linalg.norm(camera_poses[:, :3, 3] - camera_poses[0, :3, 3], axis=1).max())
        for stride in (1, 4):
            group = g[f"flow_dt{stride}"]
            assert not group["valid_mask"][-stride:].any()
            assert not group["scene_flow"][-stride:].any()
            maximum = background_max = 0.0
            count = moving = visible = background_count = 0
            for i in range(len(f["frame_index"]) - stride):
                valid = group["valid_mask"][i]
                points = g["point_map"][i]
                flow = group["scene_flow"][i]
                body = np.maximum(g["body_id"][i], 0)
                world = transform_points(points, ref)
                local = np.einsum("...ji,...j->...i", rotations[i][body], world - positions[i][body])
                endpoint_world = np.einsum("...ij,...j->...i", rotations[i + stride][body], local) + positions[i + stride][body]
                expected = transform_points(endpoint_world, inv)
                epe = np.linalg.norm(points + flow - expected, axis=-1)
                maximum = max(maximum, float(epe[valid].max(initial=0)))
                bg = valid & fixed[body]
                norm = np.linalg.norm(flow, axis=-1)
                background_max = max(background_max, float(norm[bg].max(initial=0)))
                count += int(valid.sum()); moving += int((valid & (norm > 0.001)).sum())
                visible += int(group["visible_next"][i].sum()); background_count += int(bg.sum())
            assert maximum < 3e-6, (key, stride, maximum)
            assert background_max < 1e-7, (key, stride, background_max)
            camera_report["strides"][str(stride)] = dict(max_rigid_endpoint_error_metres=maximum,
                max_static_background_flow_metres=background_max, valid_pixels=count,
                static_background_pixels=background_count, moving_pixels_over_1mm=moving,
                endpoint_visible_pixels=visible, terminal_mask="PASS")
        report["cameras"][key] = camera_report

clip = load_motion_clip(a.motion, start=a.clip_start, length=17, stride=4, normalize="clip")
xyz = np.concatenate([v["point_map"][v["valid_mask"]] for v in clip["cameras"].values()])
np.testing.assert_allclose(xyz.mean(axis=0, dtype=np.float64), 0, atol=2e-7)
assert abs(np.linalg.norm(xyz, axis=-1).mean(dtype=np.float64) - 1) < 2e-7
for v in clip["cameras"].values():
    assert not v["flow_valid_mask"][-1].any()
report["normalized_clip"] = dict(start=a.clip_start, length=17, stride=4,
    normalized_centroid=xyz.mean(axis=0, dtype=np.float64).tolist(),
    normalized_mean_radius=float(np.linalg.norm(xyz, axis=-1).mean(dtype=np.float64)),
    scale_metres=float(clip["scale"]), center=clip["center"].tolist())
payload = dict(center=clip["center"], scale=clip["scale"], frame_index=clip["frame_index"])
for key, data in clip["cameras"].items():
    for field, value in data.items(): payload[f"{key}/{field}"] = value
destination = Path(a.normalized_clip)
destination.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(destination, **payload)
report["normalized_clip"]["path"] = str(destination)
report["status"] = "PASS"
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
