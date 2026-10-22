"""Whole-episode selection, validated before loading models or parquet shards.

Dataset paths in a manifest resolve against the launch working directory, just
like dataset_dirs. Production launchers always change to the repository root.
"""
import hashlib
import json
from pathlib import Path


def load_episode_selection(path, dataset_dirs, expected_sha256=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Episode manifest SHA256 mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported episode manifest schema_version")
    mapping = manifest["episodes_by_dataset"]
    canonical = {str(Path(key).resolve()): value for key, value in mapping.items()}
    roots = [str(Path(ds).resolve()) for ds in dataset_dirs]
    if len(canonical) != len(mapping) or len(set(roots)) != len(roots):
        raise ValueError("Duplicate dataset paths in episode selection")
    if set(canonical) != set(roots):
        raise ValueError("Episode manifest must match all dataset_dirs exactly")
    selected, suites = {}, {}
    episodes = frames = 0
    for ds, root in zip(dataset_dirs, roots):
        ids = canonical[root]
        if (not isinstance(ids, list) or not ids
                or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"Expected unique nonnegative episode IDs: {ds}")
        metadata = [json.loads(line) for line in
                    (Path(root) / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        lengths = {entry["episode_index"]: entry["length"] for entry in metadata}
        if len(lengths) != len(metadata) or not set(ids).issubset(lengths):
            raise ValueError(f"Invalid episode IDs or metadata: {ds}")
        selected[str(ds)] = sorted(ids)
        nframes = sum(lengths[i] for i in ids)
        episodes += len(ids)
        frames += nframes
        suites[str(ds)] = dict(episodes=len(ids), frames=nframes)
    if (episodes != manifest.get("selected_episodes")
            or frames != manifest.get("selected_frames")):
        raise ValueError("Episode manifest counts do not match dataset metadata")
    audit = dict(sha256=digest, fraction=manifest.get("fraction"),
                 seed=manifest.get("seed"), episodes=episodes, frames=frames, suites=suites)
    return selected, audit, raw


def snapshot_episode_selection(path, dataset_dirs, output_dir, expected_sha256=None):
    _, audit, raw = load_episode_selection(path, dataset_dirs, expected_sha256)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "episode_manifest.json"
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError("Run already contains a different episode manifest")
    temporary = out / "episode_manifest.json.tmp"
    temporary.write_bytes(raw)
    temporary.replace(destination)
    (out / "episode_selection.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
