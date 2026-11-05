"""4RC labels through the unchanged flow/action sampling and processing path."""
import json
from pathlib import Path

from vpp2.libero_flow.data import FlowActionDataset
from .geometry import load_clip


class PseudoMotionReader:
    last_backend = "4rc_npz_lz4"

    def load(self, source, *, start=0, length=8, stride=4, flow_dt=4,
             prediction_horizon=32):
        if (length, stride, prediction_horizon) != (8, 4, 32) or flow_dt not in (1, 4):
            raise ValueError("4RC training requires the same 8-frame / stride4 / horizon32 sampling")
        clip = load_clip(source, start=start, dt=flow_dt)
        return dict(cameras={"image": {k: clip[k] for k in
                    ("point_map", "scene_flow", "valid_mask", "flow_valid_mask")}},
                    center=clip["center"], scale=clip["scale"], normalized=True,
                    normalization_cameras=["image"],
                    requested_frame_index=clip["frame_index"],
                    flow_target_frame_index=clip["target_frame_index"])


class PseudoFlowActionDataset(FlowActionDataset):
    def __init__(self, *args, **kwargs):
        if not kwargs.get("available_manifest"):
            raise ValueError("Supply the baseline's fixed flow manifest for a matched-source comparison")
        if kwargs.get("motion_backend") != "records":
            raise ValueError("4RC training requires LZ4 records; no HDF5 fallback")
        super().__init__(*args, **kwargs)
        if list(self.motion_cameras) != ["image"]:
            raise ValueError("4RC labels are available only for the external camera")
        expected = {tuple(x) for x in json.loads(Path(kwargs["available_manifest"]).read_text())["episodes"]}
        actual = {(r["root"].rstrip("/").split("/")[-1], r["episode"])
                  for r in self.records if r["indicator"] == 1}
        if actual != expected:
            raise ValueError(f"4RC coverage differs from fixed baseline flow population: missing={len(expected-actual)}, extra={len(actual-expected)}")
        self.motion_reader.close()
        self.motion_reader = PseudoMotionReader()
        self.audit.update(motion_source="4rc_v1", motion_storage_policy="4rc_npz_lz4",
                          flow_population="same fixed episode manifest as simulator GT experiment")

    @staticmethod
    def motion_paths(root, chunk, eid):
        directory = root / "pseudo_flow/4rc_v1" / f"episode_{eid:06d}"
        return directory, directory / "meta.json"

    @staticmethod
    def motion_complete(motion, meta, length):
        if not motion.is_dir() or not meta.is_file():
            return False
        m = json.loads(meta.read_text())
        return (m.get("complete_episode") and m.get("frames") == length
                and m.get("schema") == "4rc_material_flow_v1")
