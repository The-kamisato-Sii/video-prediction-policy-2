"""Render verified LIBERO states into dense MotionCrafter-style supervision.

Metric point maps and forward material-point displacements share the initial
agentview camera's OpenCV axes. Each camera remains a separate pixel grid.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .motion_geometry import (canonical_normalization, invert_pose, scene_flow,
                              target_visibility, transform_points, unproject)
from .motion_replay import REPLAY_VERSION

CAMERAS = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}


def set_state(model, data, state, render_data=None, frame=None):
    import mujoco

    if len(state) != 1 + model.nq + model.nv + model.na:
        raise ValueError("Saved state does not match the reconstructed MJCF")
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[-model.na:]
    mujoco.mj_forward(model, data)
    if render_data is not None:
        for key, values in render_data.items():
            if key.startswith("model_"):
                getattr(model, key.removeprefix("model_"))[:] = values[frame]
            else:
                getattr(data, key)[:] = values[frame]


def camera_pose(data, camera_id):
    pose = np.eye(4)
    # MuJoCo: right/up/backward. OpenCV: right/down/forward.
    pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    pose[:3, 3] = data.cam_xpos[camera_id]
    return pose


def decode_rgb(path):
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def orient(array, name):
    if name == "identity":
        return array
    if name == "flip_y":
        return array[::-1]
    if name == "flip_x":
        return array[:, ::-1]
    if name == "rotate180":
        return array[::-1, ::-1]
    raise ValueError(name)


def extract(dataset, episode, replay_path, resolution=224, strides=(1, 4),
            max_rgb_mae=3.0, max_frames=None, output_root=None):
    import h5py
    import mujoco

    if mujoco.__version__ != "3.3.2":
        raise ValueError("MuJoCo 3.3.2 is required for source-RGB alignment")
    if resolution <= 0 or not strides or min(strides) < 1:
        raise ValueError("Positive resolution and flow strides required")
    dataset = Path(dataset)
    parquet = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    with h5py.File(replay_path, "r") as f:
        if not f.attrs.get("replay_verified", False):
            raise ValueError("First run motion_replay with strict proprio agreement")
        if f.attrs.get("replay_version", 1) != REPLAY_VERSION:
            raise ValueError("Replay cache predates the fixture reset-order correction")
        if f.attrs["parquet_sha256"] != hashlib.sha256(parquet.read_bytes()).hexdigest():
            raise ValueError("Replay belongs to another LeRobot episode")
        xml = f.attrs["model_file"]
        states = f["states"][:]
        image_sim_time = f["image_sim_time"][:]
        source_indices = f["source_indices"][:]
        replay_attrs = dict(f.attrs)
        render_data = {key: values[:] for key, values in f.get("render_data", {}).items()}
        replay_proprio_error = f["proprio_errors"][:].max(axis=0).tolist()
    if max_frames:
        states = states[:max_frames]
        image_sim_time = image_sim_time[:max_frames]
        source_indices = source_indices[:max_frames]
    count = len(states)
    rgb = {key: decode_rgb(dataset / "videos" / f"chunk-{episode // 1000:03d}" /
                           f"observation.images.{key}" / f"episode_{episode:06d}.mp4")[:count]
           for key in CAMERAS}
    if any(len(v) != count for v in rgb.values()):
        raise ValueError("Video frame count does not match retained simulator states")
    native = rgb["image"].shape[1]
    if any(v.shape[1:3] != (native, native) for v in rgb.values()):
        raise ValueError("Expected square native camera images")
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = max(native, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(native, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera_ids = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for k, name in CAMERAS.items()}
    if min(camera_ids.values()) < 0:
        raise ValueError("MJCF is missing a dataset camera")
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.geomgroup[1] = 1
    set_state(model, data, states[0], render_data, 0)
    reference = camera_pose(data, camera_ids["image"])
    world_to_reference = invert_pose(reference)
    rotations, positions = [], []
    poses = {key: [] for key in CAMERAS}
    for i, state in enumerate(states):
        set_state(model, data, state, render_data, i)
        rotations.append(data.xmat.reshape(-1, 3, 3).copy())
        positions.append(data.xpos.copy())
        for key, cid in camera_ids.items():
            poses[key].append(camera_pose(data, cid))
    rotations, positions = np.asarray(rotations), np.asarray(positions)
    out_root = Path(output_root) if output_root else dataset
    output = out_root / "motions" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.hdf5"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    report = dict(episode=episode, frames=count, resolution=resolution, native_resolution=native,
                  flow_strides=list(strides), source_rgb_mae={}, camera_orientation={},
                  coordinate_frame="initial_agentview_opencv", units="metres", complete_episode=max_frames is None)
    report["max_replay_proprio_error"] = dict(joint_rad=replay_proprio_error[0], gripper_metres=replay_proprio_error[1])
    report["replay_version"] = REPLAY_VERSION
    report["reset_count"] = int(replay_attrs["reset_count"])
    report["render_capture"] = replay_attrs["render_capture"]
    try:
        with mujoco.Renderer(model, height=native, width=native) as renderer, h5py.File(temporary, "w") as out:
            out.attrs["schema_version"] = 1
            out.attrs["replay_version"] = REPLAY_VERSION
            out.attrs["coordinate_frame"] = "initial_agentview_opencv"
            out.attrs["normalization"] = "MotionCrafter sequence center and mean radius; flow divided by radius only"
            out.attrs["replay_sha256"] = hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()
            out.attrs["parquet_sha256"] = replay_attrs["parquet_sha256"]
            out.attrs["source_url"] = replay_attrs.get("source_source_url", replay_attrs["source"])
            out.attrs["source_demo"] = replay_attrs["demo"]
            out.attrs["source_indices_note"] = "original demo indices; physical timing follows replayed, no-op-filtered controls"
            out.create_dataset("frame_index", data=np.arange(count))
            out.create_dataset("timestamp", data=np.arange(count) / 20)
            out.create_dataset("image_sim_time", data=image_sim_time)
            out.create_dataset("source_frame_index", data=source_indices)
            out.create_dataset("reference_camera_to_sim_world", data=reference)
            out.create_dataset("body_rotations", data=rotations)
            out.create_dataset("body_positions", data=positions)
            for key, cid in camera_ids.items():
                group = out.create_group(key)
                h = w = resolution
                chunks = (1, h, w)
                for name, dtype, tail in [("point_map", "f4", (3,)), ("depth", "f4", ()),
                                           ("valid_mask", "?", ()), ("body_id", "i4", ()),
                                           ("geom_id", "i4", ()), ("rgb", "u1", (3,))]:
                    group.create_dataset(name, shape=(count, h, w) + tail,
                                         dtype=dtype, chunks=chunks + tail,
                                         compression="gzip", compression_opts=1, shuffle=True)
                frame_errors = []
                orientation = None
                ys = np.minimum(((np.arange(h) + 0.5) * native / h).astype(int), native - 1)
                xs = np.minimum(((np.arange(w) + 0.5) * native / w).astype(int), native - 1)
                focal = native / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
                intrinsic = np.array([[focal, 0, (native - 1) / 2], [0, focal, (native - 1) / 2], [0, 0, 1]])
                group.create_dataset("intrinsic_native", data=intrinsic)
                group.create_dataset("sample_pixel_x", data=xs)
                group.create_dataset("sample_pixel_y", data=ys)
                for i, state in enumerate(states):
                    set_state(model, data, state, render_data, i)
                    renderer.disable_depth_rendering()
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(data, camera=cid, scene_option=options)
                    rendered = renderer.render().copy()
                    if orientation is None:
                        errors = {name: float(np.abs(orient(rendered, name).astype(float) - rgb[key][i]).mean())
                                  for name in ("identity", "flip_y", "flip_x", "rotate180")}
                        orientation = min(errors, key=errors.get)
                        print(key, "orientation RGB errors", errors, flush=True)
                    mae = float(np.abs(orient(rendered, orientation).astype(float) - rgb[key][i]).mean())
                    frame_errors.append(mae)
                    if mae > max_rgb_mae:
                        raise ValueError(f"RGB alignment failed: {key} frame {i}, MAE {mae:.3f} > {max_rgb_mae}; no labels published")
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    geom = segmentation[..., 0]
                    valid = ((segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                             (geom >= 0) & np.isfinite(depth) & (depth > 0) & (depth < 10))
                    body = model.geom_bodyid[np.clip(geom, 0, model.ngeom - 1)]
                    point_world = unproject(depth, intrinsic, poses[key][i])
                    point = transform_points(point_world, world_to_reference)
                    point[~valid] = 0
                    arrays = dict(point_map=point, depth=depth, valid_mask=valid,
                                  body_id=np.where(valid, body, -1), geom_id=np.where(valid, geom, -1), rgb=rendered)
                    for name, array in arrays.items():
                        group[name][i] = orient(array, orientation)[ys[:, None], xs[None, :]]
                    if i % 25 == 0:
                        print(f"Render {key} {i}/{count} RGB_MAE={mae:.3f}", flush=True)
                group.attrs["image_orientation"] = orientation
                group.create_dataset("camera_to_reference", data=[world_to_reference @ p for p in poses[key]])
                group.create_dataset("rgb_mae", data=frame_errors)
                report["camera_orientation"][key] = orientation
                report["source_rgb_mae"][key] = dict(mean=float(np.mean(frame_errors)), max=float(np.max(frame_errors)))
                # Sampling an already oriented native pixel grid is represented
                # explicitly; downstream code need not guess a resized K.
                for stride in strides:
                    fg = group.create_group(f"flow_dt{stride}")
                    fg.attrs["dataset_step_delta"] = stride
                    fg.attrs["nominal_seconds"] = stride / 20
                    for name, dtype, tail in [("scene_flow", "f4", (3,)), ("valid_mask", "?", ()), ("visible_next", "?", ())]:
                        fg.create_dataset(name, shape=(count, h, w) + tail,
                                          chunks=chunks + tail, dtype=dtype,
                                          compression="gzip", compression_opts=1, shuffle=True)
                    for i in range(count - stride):
                        valid = group["valid_mask"][i]
                        source_points = group["point_map"][i]
                        world = transform_points(source_points, reference)
                        body = group["body_id"][i]
                        _, flow, moved = scene_flow(world, body,
                            (rotations[i], positions[i]), (rotations[i + stride], positions[i + stride]), reference, valid)
                        fg["scene_flow"][i] = flow
                        fg["valid_mask"][i] = valid
                        # Endpoint visibility is computed on native target pixels,
                        # with exact rigid advection, including out-of-view points.
                        set_state(model, data, states[i + stride], render_data, i + stride)
                        renderer.update_scene(data, camera=cid, scene_option=options)
                        renderer.enable_depth_rendering()
                        target_depth = renderer.render().copy()
                        renderer.disable_depth_rendering()
                        fg["visible_next"][i] = target_visibility(moved, target_depth, intrinsic, poses[key][i + stride], valid)
                    # Last stride rows remain zero with false validity masks.
            all_points = [out[k]["point_map"] for k in CAMERAS]
            all_masks = [out[k]["valid_mask"] for k in CAMERAS]
            # Flatten camera/time into a sequence without materializing every map.
            class Combined:
                def __init__(self, values): self.values = values
                def __iter__(self):
                    for values in self.values:
                        yield from values
            center, scale = canonical_normalization(Combined(all_points), Combined(all_masks))
            out.create_dataset("normalization/center", data=center)
            out.create_dataset("normalization/scale", data=scale)
            report["normalization"] = dict(center=center.tolist(), scale_metres=float(scale), scope="entire_episode_both_cameras")
            out.attrs["report_json"] = json.dumps(report)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    meta = out_root / "meta" / "motions"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"episode_{episode:06d}.json").write_text(json.dumps(report, indent=2) + "\n")
    (meta / "schema.json").write_text(json.dumps(dict(
        version=1, path="motions/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.hdf5",
        coordinate_frame="first agentview camera, OpenCV axes; shared by both views",
        point_map="metric xyz at source pixels", scene_flow="same rigid material point displacement in metres",
        normalization="Xn=(X-center)/scale; Vn=V/scale; no clipping",
        terminal="last dt rows have zero flow and false valid_mask",
        cameras=CAMERAS), indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 4])
    p.add_argument("--max-rgb-mae", type=float, default=3.0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--output-root", help="Alternate root for bounded probes")
    a = p.parse_args()
    print(extract(a.dataset, a.episode, a.replay, a.resolution, a.strides,
                  a.max_rgb_mae, a.max_frames, a.output_root))


if __name__ == "__main__":
    main()
