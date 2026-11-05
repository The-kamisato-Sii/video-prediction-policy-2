"""All-episode flow plus subset action labels, packed as separate task samples."""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from torch.utils.data import Dataset, default_collate

from vpp2.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from vpp2.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames_pyav
from .motion_reader import MotionReader
from .views import DEFAULT_MOTION_CAMERAS, motion_cameras

CAMERAS = ("image", "wrist_image")


class FlowActionDataset(Dataset):
    def __init__(self, cfg, denominator=8, flow_dt=4, action_only=False,
                 missing_motion="skip", available_manifest=None,
                 motion_backend="auto", motion_open_files=8, flow_episode_scope="all",
                 cameras=DEFAULT_MOTION_CAMERAS):
        if flow_episode_scope not in ("all", "complementary"):
            raise ValueError("flow_episode_scope must be all or complementary")
        if denominator not in (1, 2, 4, 8) or flow_dt not in (1, 4):
            raise ValueError("Expected denominator1/2/4/8 and flow_dt1/4")
        self.cfg, self.denominator, self.flow_dt = cfg, denominator, flow_dt
        self.motion_cameras = motion_cameras(cameras)
        self.motion_reader = MotionReader(motion_backend, motion_open_files,
                                          normalization_cameras=self.motion_cameras)
        self.processor = instantiate(cfg.processor)
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.pretrained_norm_stats))
        self.processor.train()
        manifest_path = Path(f"configs/libero/subsets/episodes_1of{denominator}_seed42.json")
        manifest = json.loads(manifest_path.read_text())
        selected = {Path(k).name: set(v) for k, v in manifest["episodes_by_dataset"].items()}
        available = None
        if available_manifest:
            available = {tuple(x) for x in json.loads(Path(available_manifest).read_text())["episodes"]}
        self.episodes, self.ends, self.records = [], [], []
        self._tables, self._texts = OrderedDict(), {}
        missing = []
        total = 0
        for directory in cfg.dataset_dirs:
            root = Path(directory)
            info = json.loads((root / "meta/info.json").read_text())
            tasks = {r["task_index"]: r["task"] for r in self._jsonl(root / "meta/tasks.jsonl")}
            episodes = self._jsonl(root / "meta/episodes.jsonl")
            for ep in episodes:
                eid, length = int(ep["episode_index"]), int(ep["length"])
                # Selected episodes contribute TWO independent task records:
                # an action-labelled view and a flow-labelled view. Neither
                # branch reads the other task's labels.
                modes = [0] if eid in selected[root.name] else []
                if not action_only and (flow_episode_scope == "all" or not modes):
                    modes.append(1)
                for indicator in modes:
                    chunk = eid // int(info["chunks_size"])
                    motion, meta = self.motion_paths(root, chunk, eid)
                    if indicator:
                        valid = (available is None or (root.name, eid) in available) and self.motion_complete(motion, meta, length)
                        if not valid:
                            missing.append([root.name, eid])
                            continue
                    # Keep baseline action starts, including padded tails. Flow starts
                    # require all8 endpoints to exist for BOTH dt settings.
                    starts = length if indicator == 0 else max(length - 32, 0)
                    record = dict(root=str(root), episode=eid, length=length, indicator=indicator,
                                  chunk=chunk, fps=info["fps"], tasks=tasks, motion=str(motion), starts=starts)
                    self.records.append(record)
                    if starts:
                        self.episodes.append(record)
                        total += starts
                        self.ends.append(total)
        if missing and missing_motion == "error":
            raise FileNotFoundError(f"Incomplete flow coverage: {len(missing)} episodes; examples={missing[:8]}. Use missing_motion=skip to omit only their flow records.")
        if missing_motion not in ("error", "skip"):
            raise ValueError(missing_motion)
        self.audit = dict(action_fraction=f"1/{denominator}", flow_dt=flow_dt,
                          motion_cameras=list(self.motion_cameras), rgb_cameras=list(CAMERAS),
                          motion_normalization="selected_views_clip_mean_radius_v1",
                          motion_normalization_cameras=list(self.motion_cameras),
                          action_episodes=sum(r["indicator"] == 0 for r in self.records),
                          flow_episodes=sum(r["indicator"] == 1 for r in self.records),
                          action_starts=sum(r["starts"] for r in self.records if r["indicator"] == 0),
                          flow_starts=sum(r["starts"] for r in self.records if r["indicator"] == 1),
                          missing_flow_episodes=missing, action_only=bool(action_only),
                          manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                          available_manifest_sha256=hashlib.sha256(Path(available_manifest).read_bytes()).hexdigest() if available_manifest else None,
                          flow_episode_scope=flow_episode_scope,
                          partition=f"whole-episode action subset; {flow_episode_scope} flow episodes; uniform task-start sampling",
                          tail_policy="baseline padded action starts retained; flow requires start+32<length")
        action_ids = {(r["root"], r["episode"]) for r in self.records if r["indicator"] == 0}
        flow_ids = {(r["root"], r["episode"]) for r in self.records if r["indicator"] == 1}
        self.audit["overlapping_action_flow_episodes"] = len(action_ids & flow_ids)
        self.audit["missing_motion_policy"] = missing_motion
        self.audit["skipped_flow_episodes"] = len(missing) if missing_motion == "skip" else 0
        self.audit["action_episodes_without_flow"] = len(action_ids - flow_ids) if not action_only else 0
        self.audit["sampled_action_fraction"] = self.audit["action_starts"] / total if total else 0.0
        self.audit["motion_storage_policy"] = motion_backend
        if not self.ends:
            raise ValueError("Empty dataset")

    @staticmethod
    def motion_paths(root, chunk, eid):
        return (root / f"motions/chunk-{chunk:03d}/episode_{eid:06d}.hdf5",
                root / f"meta/motions/episode_{eid:06d}.json")

    @staticmethod
    def motion_complete(motion, meta, length):
        if not motion.is_file() or not meta.is_file():
            return False
        m = json.loads(meta.read_text())
        return m.get("replay_version") == 3 and m.get("complete_episode") and m.get("frames") == length

    @staticmethod
    def _jsonl(path):
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        i = bisect.bisect_right(self.ends, int(index))
        return self.episodes[i], int(index) - (self.ends[i - 1] if i else 0)

    def first_index(self, indicator, suite=None):
        for i, r in enumerate(self.episodes):
            if r["indicator"] == indicator and (suite is None or suite in r["root"]):
                return self.ends[i-1] if i else 0
        raise ValueError(f"No samples for indicator={indicator}, suite={suite}")

    def _table(self, r):
        key = (r["root"], r["episode"], r["indicator"])
        if key not in self._tables:
            columns = ["observation.state", "timestamp", "task_index", "frame_index"]
            if r["indicator"] == 0:
                columns.append("action")
            path = Path(r["root"]) / f"data/chunk-{r['chunk']:03d}/episode_{r['episode']:06d}.parquet"
            # A real parquet column projection; never load hidden actions and drop later.
            table = pq.read_table(path, columns=columns).to_pydict()
            self._tables[key] = (table, tuple(columns))
            if len(self._tables) > 8:
                self._tables.popitem(last=False)
        self._tables.move_to_end(key)
        return self._tables[key]

    def __getitem__(self, index):
        r, start = self.locate(index)
        table, columns = self._table(r)
        if int(table["frame_index"][start]) != start:
            raise ValueError("Non-contiguous retained frame numbering")
        raw = dict(idx=int(index), task=r["tasks"][int(table["task_index"][start])],
                   state={"default": torch.tensor([table["observation.state"][start]], dtype=torch.float32)},
                   state_is_pad=torch.zeros(1, dtype=torch.bool),
                   image_is_pad=torch.zeros(1, dtype=torch.bool), images={})
        if r["indicator"] == 0:
            requested = np.arange(start, start + 32)
            ids = np.minimum(requested, r["length"] - 1)
            raw["action"] = {"default": torch.tensor(np.asarray(table["action"], dtype=np.float32)[ids].copy())}
            raw["action_is_pad"] = torch.from_numpy(requested >= r["length"])
        for camera in CAMERAS:
            path = Path(r["root"]) / f"videos/chunk-{r['chunk']:03d}/observation.images.{camera}/episode_{r['episode']:06d}.mp4"
            raw["images"][camera] = decode_video_frames_pyav(path, [table["timestamp"][start]], 1e-4, return_uint8=True)
        item = self.processor.preprocess(raw)
        # Identical Resize/ToTensor per view and horizontal layout as baseline.
        video = torch.cat(list(item["pixel_values"].unbind(0)), dim=-1)
        video = ((video - 0.5) / 0.5).permute(1, 0, 2, 3)
        prompt = DEFAULT_PROMPT.format(task=item["instruction"])
        if prompt not in self._texts:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            path = Path(self.cfg.text_embedding_cache_dir) / f"{digest}.t5_len{self.cfg.context_len}.{self.cfg.text_embedding_cache_tag}.pt"
            text = torch.load(path, map_location="cpu", weights_only=True)
            context = text["context"].clone()
            context[~text["mask"].bool()] = 0
            self._texts[prompt] = (context, torch.ones_like(text["mask"], dtype=torch.bool))
        context, context_mask = self._texts[prompt]
        result = dict(video=video, video_num_frames=17, context=context.clone(),
                      context_mask=context_mask.clone(), proprio=item["proprio"][:1],
                      image_is_pad=item["image_is_pad"], indicator=r["indicator"], prompt=prompt,
                      provenance=dict(dataset=r["root"], episode=r["episode"], start=start,
                                      parquet_columns=list(columns), motion_opened=bool(r["indicator"])))
        if r["indicator"] == 0:
            result.update(action=item["action"], action_is_pad=item["action_is_pad"])
        else:
            clip = self.motion_reader.load(r["motion"], start=start, length=8, stride=4,
                                           flow_dt=self.flow_dt, prediction_horizon=32)
            result["provenance"]["motion_storage"] = self.motion_reader.last_backend
            maps, masks = [], []
            result["provenance"]["motion_cameras"] = list(self.motion_cameras)
            result["provenance"]["motion_normalization_cameras"] = clip["normalization_cameras"]
            for camera in self.motion_cameras:
                c = clip["cameras"][camera]
                maps.append(np.concatenate([c["point_map"], c["scene_flow"]], axis=-1))
                masks.append(np.stack([c["valid_mask"], c["flow_valid_mask"]], axis=-1))
            result["motion"] = torch.from_numpy(np.stack(maps)).permute(0, 4, 1, 2, 3).contiguous()
            result["motion_valid"] = torch.from_numpy(np.stack(masks)).permute(0, 4, 1, 2, 3).contiguous()
            result["motion_frame_index"] = torch.from_numpy(clip["requested_frame_index"])
            result["flow_target_frame_index"] = torch.from_numpy(clip["flow_target_frame_index"])
            result["motion_center"] = torch.as_tensor(clip["center"], dtype=torch.float32)
            result["motion_scale"] = torch.tensor(clip["scale"], dtype=torch.float32)
        return result


def collate_flow_action(samples):
    """Pack only observed labels; never fabricate absent targets."""
    common = ("video", "video_num_frames", "context", "context_mask", "proprio", "image_is_pad", "indicator")
    batch = {k: default_collate([s[k] for s in samples]) for k in common}
    batch["provenance"] = [s["provenance"] for s in samples]
    for indicator, name, keys in ((0, "action", ("action", "action_is_pad")),
                                   (1, "flow", ("motion", "motion_valid", "motion_frame_index", "flow_target_frame_index", "motion_center", "motion_scale"))):
        ids = [i for i, s in enumerate(samples) if s["indicator"] == indicator]
        batch[f"{name}_indices"] = torch.tensor(ids, dtype=torch.long)
        if ids:
            for k in keys:
                batch[k] = default_collate([samples[i][k] for i in ids])
    return batch
