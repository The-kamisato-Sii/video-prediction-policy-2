"""Replay a matched original demo and require agreement with LeRobot joints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .motion_source import retained_action_indices

REPLAY_VERSION = 3


def replay(dataset, episode, source_snapshot, output, settle_steps=10, joint_tolerance=1e-5):
    import h5py
    import pyarrow.parquet as pq
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    import robosuite

    if mujoco.__version__ != "3.3.2" or robosuite.__version__ != "1.4.0":
        raise ValueError("This data release requires MuJoCo 3.3.2 and robosuite 1.4.0")
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(output)
    parquet = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    table = pq.read_table(parquet).to_pydict()
    joints = np.asarray(table["observation.states.joint_state"])
    gripper = np.asarray(table["observation.states.gripper_state"])
    with h5py.File(source_snapshot, "r") as f:
        if f.attrs["parquet_sha256"] != digest:
            raise ValueError("Source snapshot belongs to a different LeRobot episode")
        initial = f["original_initial_state"][:]
        indices = f["source_indices"][:]
        original_actions = f["original_actions"][:]
        if not np.array_equal(indices, retained_action_indices(original_actions)):
            raise ValueError("Source indices disagree with the original no-op filtering rule")
        actions = original_actions[indices]
        source_attrs = dict(f.attrs)
    bddl = Path(get_libero_path("bddl_files")) / Path(source_attrs["source_bddl_file_name"]).parent.name / Path(source_attrs["source_bddl_file_name"]).name
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=512,
                            camera_widths=512, use_camera_obs=True)
    try:
        env.seed(0)
        # Fixed fixture poses live in model.body_pos/body_quat, not qpos.
        # Regeneration seeded one environment per task and reset it once per
        # ORIGINAL demo, including demos later discarded as unsuccessful.
        reset_count = int(source_attrs["demo"].split("_")[-1]) + 1
        for _ in range(reset_count):
            env.reset()
        # Robosuite samples cameras one 2 ms integration step before the end
        # of each 50 ms control step. Capture transforms INSIDE the camera
        # callback, not after env.step(). Rendering itself does not advance
        # physics; placeholders avoid rendering RGB twice (extract does it).
        render_fields = ("xpos", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat",
                         "site_xpos", "site_xmat", "light_xpos", "light_xdir", "subtree_com")
        model_fields = ("site_rgba", "geom_rgba", "mat_rgba")
        captures = {}

        def capture_camera(*args, **kwargs):
            name = kwargs.get("camera_name")
            if args or name not in ("agentview", "robot0_eye_in_hand") or kwargs.get("depth") or kwargs.get("segmentation"):
                raise ValueError("Unexpected camera callback contract")
            snapshot = {key: np.asarray(getattr(env.sim.data, key)).copy() for key in render_fields}
            snapshot.update({"model_" + key: np.asarray(getattr(env.sim.model, key)).copy() for key in model_fields})
            captures[name] = (float(env.sim.data.time), env.sim.get_state().flatten().copy(), snapshot)
            return np.zeros((kwargs["height"], kwargs["width"], 3), dtype=np.uint8)

        env.sim.render = capture_camera
        obs = env.set_init_state(initial)
        for _ in range(settle_steps):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        states, errors, image_times = [], [], []
        # mj_step integrates qpos after updating render transforms. Rendering
        # saved qpos with a fresh mj_forward can shift fast-moving surfaces by
        # one internal simulator step. Preserve the actual image-time transforms.
        render_data = {key: [] for key in render_fields + tuple("model_" + k for k in model_fields)}
        for i, action in enumerate(actions):
            je = float(np.max(np.abs(obs["robot0_joint_pos"] - joints[i])))
            ge = float(np.max(np.abs(obs["robot0_gripper_qpos"] - gripper[i])))
            errors.append((je, ge))
            if max(je, ge) > joint_tolerance:
                raise ValueError(f"Replay mismatch at frame {i}: joint={je:.9g}, gripper={ge:.9g}; no motion labels written")
            left, right = captures["agentview"], captures["robot0_eye_in_hand"]
            if left[0] != right[0] or not np.array_equal(left[1], right[1]):
                raise ValueError("The two cameras were sampled at different simulation times")
            image_times.append(left[0])
            states.append(left[1])
            for key in render_data:
                render_data[key].append(left[2][key])
            obs, _, _, _ = env.step(action)
        xml = env.sim.model.get_xml()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        with h5py.File(temporary, "w") as f:
            for k, v in source_attrs.items():
                f.attrs[k] = v
            f.attrs["model_file"] = xml
            f.attrs["replay_verified"] = True
            f.attrs["replay_version"] = REPLAY_VERSION
            f.attrs["reset_count"] = reset_count
            f.attrs["noop_rule"] = "arm_norm_lt_1e-4_and_same_last_retained_gripper"
            f.attrs["render_capture"] = "camera_sensor_callback_before_post_action"
            f.attrs["settle_steps"] = settle_steps
            f.attrs["joint_tolerance"] = joint_tolerance
            f.attrs["mujoco_version"] = mujoco.__version__
            f.attrs["robosuite_version"] = robosuite.__version__
            f.attrs["source_snapshot_sha256"] = hashlib.sha256(Path(source_snapshot).read_bytes()).hexdigest()
            f.create_dataset("states", data=states)
            f.create_dataset("image_sim_time", data=image_times)
            f.create_dataset("source_indices", data=indices)
            f.create_dataset("proprio_errors", data=errors)
            for key, values in render_data.items():
                f.create_dataset(f"render_data/{key}", data=values)
        temporary.replace(output)
        print(json.dumps(dict(frames=len(states), max_joint_error=np.max(errors, axis=0).tolist(), output=str(output)), indent=2), flush=True)
    finally:
        env.close()
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--source-snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--joint-tolerance", type=float, default=1e-5)
    a = p.parse_args()
    replay(a.dataset, a.episode, a.source_snapshot, a.output, a.settle_steps, a.joint_tolerance)


if __name__ == "__main__":
    main()
