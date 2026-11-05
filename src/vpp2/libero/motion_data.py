"""Read normalized motion clips without changing the existing action loader."""
import numpy as np

from .motion_geometry import canonical_normalization
from .motion_replay import REPLAY_VERSION


def load_motion_clip(path, start=0, length=17, stride=4, normalize="clip", *,
                     flow_dt=None, prediction_horizon=None):
    """Return two views in the same fixed frame with explicit target bounds.

    Metric data are retained on disk. ``clip`` follows MotionCrafter's sequence
    normalization on this exact clip; ``episode`` uses the saved episode scale.
    Ground-truth-derived normalization parameters belong to targets only.

    ``stride`` selects source frames; ``flow_dt`` independently selects their
    displacement interval (defaults to ``stride`` for existing callers).
    ``prediction_horizon`` is the inclusive target endpoint offset from
    ``start``, in retained dataset steps. It defaults to the last source offset,
    preserving the old terminal-flow mask. For eight sources 0,4,...,28 aligned
    with 32 actions, pass length=8, stride=4, prediction_horizon=32 and flow_dt=1
    or 4. The eighth flow then remains valid when its endpoint exists.
    """
    import h5py

    if length < 1 or stride < 1 or start < 0:
        raise ValueError("Invalid clip sampling")
    flow_dt = stride if flow_dt is None else flow_dt
    last_source_offset = (length - 1) * stride
    prediction_horizon = (last_source_offset if prediction_horizon is None
                          else prediction_horizon)
    if flow_dt < 1:
        raise ValueError("flow_dt must be positive")
    if prediction_horizon < last_source_offset:
        raise ValueError("prediction_horizon cannot precede the last source frame")
    with h5py.File(path, "r") as f:
        if f.attrs.get("replay_version", 1) != REPLAY_VERSION:
            raise ValueError("Motion labels predate corrected fixture/camera replay; regenerate them")
        n = len(f["frame_index"])
        if start >= n:
            raise IndexError(start)
        requested = start + np.arange(length) * stride
        indices = np.minimum(requested, n - 1)
        valid_time = requested < n
        target_indices = requested + flow_dt
        valid_flow_time = (valid_time & (target_indices < n)
                           & (target_indices <= start + prediction_horizon))
        cameras = {}
        for key in ("image", "wrist_image"):
            g = f[key]
            # Explicit indexing preserves duplicated tail frames with HDF5.
            points = np.stack([g["point_map"][i] for i in indices])
            flow = np.stack([g[f"flow_dt{flow_dt}/scene_flow"][i] for i in indices])
            valid = np.stack([g["valid_mask"][i] for i in indices]) & valid_time[:, None, None]
            flow_valid = np.stack([g[f"flow_dt{flow_dt}/valid_mask"][i] for i in indices])
            flow_valid &= valid_flow_time[:, None, None]
            cameras[key] = dict(point_map=points, scene_flow=flow, valid_mask=valid, flow_valid_mask=flow_valid)
        if normalize == "clip":
            center, scale = canonical_normalization(
                [v["point_map"] for v in cameras.values()], [v["valid_mask"] for v in cameras.values()])
        elif normalize == "episode":
            center, scale = f["normalization/center"][:], float(f["normalization/scale"][()])
        elif normalize == "none":
            center, scale = np.zeros(3), 1.0
        else:
            raise ValueError(normalize)
        for v in cameras.values():
            v["point_map"] = ((v["point_map"] - center) / scale).astype(np.float32)
            v["scene_flow"] = (v["scene_flow"] / scale).astype(np.float32)
            v["point_map"][~v["valid_mask"]] = 0
            v["scene_flow"][~v["flow_valid_mask"]] = 0
        return dict(cameras=cameras, frame_index=indices, requested_frame_index=requested,
                    flow_target_frame_index=target_indices, sample_stride=stride,
                    flow_dt=flow_dt, prediction_horizon=prediction_horizon,
                    center=center, scale=scale, normalized=normalize != "none")
