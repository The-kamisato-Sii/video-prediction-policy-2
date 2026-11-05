"""Add verified motion sidecars to selected existing LeRobot episodes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .motion_source import recover_episode
from .motion_replay import replay, REPLAY_VERSION
from .motion_extract import extract


def source_filename(suite, language):
    """Resolve scene-prefixed LIBERO-10 filenames from the task definitions."""
    stem = language.replace(" ", "_")
    root = Path(__file__).resolve().parents[3]
    candidates = [p for p in (root / "third_party/LIBERO-PRO-assets/libero/libero/bddl_files" / suite).glob("*.bddl")
                  if p.stem == stem or p.stem.endswith("_" + stem)]
    if len(candidates) != 1:
        raise ValueError(f"Cannot uniquely resolve {suite}: {language}")
    return candidates[0].stem + "_demo.hdf5"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument("--episodes", type=int, nargs="+")
    select.add_argument("--all-episodes", action="store_true")
    p.add_argument("--source-dir", help="Optional local folder of original task HDF5s")
    p.add_argument("--cache-dir", default="data/libero_motion_sources")
    p.add_argument("--source-revision", default="f13aa24a3da8c43c7225569f28c562979fa0e35a")
    p.add_argument("--resolution", type=int, default=224)
    a = p.parse_args()
    import h5py

    dataset, cache = Path(a.dataset), Path(a.cache_dir)
    while (cache / "validation_in_progress").exists():
        print("Waiting for motion validation to finish", flush=True)
        time.sleep(30)
    entries = {e["episode_index"]: e for e in map(json.loads, (dataset / "meta/episodes.jsonl").read_text().splitlines())}
    episodes = sorted(entries) if a.all_episodes else a.episodes
    suite = dataset.name.removesuffix("_no_noops_lerobot")
    for episode in episodes:
        entry = entries[episode]
        if len(entry["tasks"]) != 1:
            raise ValueError("Expected one task per episode")
        destination = dataset / "motions" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.hdf5"
        if destination.exists():
            parquet = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
            with h5py.File(destination, "r") as f:
                version = f.attrs.get("replay_version", 1)
                report = json.loads(f.attrs["report_json"])
                if (f.attrs["parquet_sha256"] != hashlib.sha256(parquet.read_bytes()).hexdigest()
                        or not report["complete_episode"] or report["resolution"] != a.resolution):
                    raise ValueError(f"Existing motion sidecar has a different contract: {destination}")
            if version == REPLAY_VERSION:
                print("Already complete:", destination, flush=True)
                continue
            archived = dataset / "motions" / f"legacy_replay_v{version}" / destination.parent.name / destination.name
            archived.parent.mkdir(parents=True, exist_ok=True)
            if archived.exists():
                raise FileExistsError(archived)
            destination.rename(archived)
            metadata = dataset / "meta/motions" / f"episode_{episode:06d}.json"
            if metadata.exists():
                old_meta = metadata.parent / f"legacy_replay_v{version}" / metadata.name
                old_meta.parent.mkdir(parents=True, exist_ok=True)
                metadata.rename(old_meta)
            print("Archived old reset-order labels:", archived, flush=True)
        filename = source_filename(suite, entry["tasks"][0])
        source = str(Path(a.source_dir) / filename) if a.source_dir else (
            f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{a.source_revision}/{suite}/{filename}?download=true")
        existing = cache / f"{dataset.name}_episode_{episode:06d}.hdf5"
        if existing.exists():
            with h5py.File(existing, "r") as f:
                bddl = Path(f.attrs["source_bddl_file_name"])
                if bddl.parent.name != suite or bddl.stem + "_demo.hdf5" != filename:
                    raise ValueError("Cached original source belongs to another task")
                source = str(f.attrs["source"])
        snapshot = recover_episode(dataset, episode, source, cache)
        trajectory = cache / f"replayed_v{REPLAY_VERSION}_{dataset.name}_episode_{episode:06d}.hdf5"
        if not trajectory.exists():
            replay(dataset, episode, snapshot, trajectory)
        extract(dataset, episode, trajectory, resolution=a.resolution)


if __name__ == "__main__":
    main()
