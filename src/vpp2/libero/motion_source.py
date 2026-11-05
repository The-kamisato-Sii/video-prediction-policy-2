"""Recover original simulator states by matching a LeRobot action sequence.

HTTP range reads avoid downloading the RGB arrays in the original LIBERO HDF5.
Only an exact, monotonic action match is accepted; episode numbering is never
assumed to survive filtering. Joint positions provide an independent check.
"""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only HTTP file with checked byte ranges and a disk cache."""

    def __init__(self, url, cache_dir, block_size=262144):
        import requests

        self.url = url
        self.session = requests.Session()
        self.block_size = block_size
        self.cache = Path(cache_dir) / hashlib.sha256(url.encode()).hexdigest()[:20]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.bytes_downloaded = 0
        self.blocks = OrderedDict()
        info = self.cache / "http.json"
        if info.exists():
            self.size = json.loads(info.read_text())["size"]
        else:
            data, content_range = self._request(0, 0)
            if not content_range.startswith("bytes 0-0/") or len(data) != 1:
                raise ValueError("Source must support HTTP byte ranges")
            self.size = int(content_range.split("/")[-1])
            info.write_text(json.dumps(dict(url=url, size=self.size)))

    def _request(self, start, end):
        separator = "&" if "?" in self.url else "?"
        url = f"{self.url}{separator}motion_range={start}-{end}"
        try:
            r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 40))
            r.raise_for_status()
            if r.status_code != 206:
                raise ValueError("HTTP server ignored range")
            return r.content, r.headers.get("Content-Range", "")
        except Exception:
            # Some cluster images have a different TLS/proxy stack in requests
            # and curl. Keep certificate verification enabled in either path.
            with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
                body, header = Path(tmp) / "body", Path(tmp) / "header"
                subprocess.run(["curl", "--silent", "--show-error", "--location", "--fail",
                                "--retry", "2", "--max-time", "120", "--range", f"{start}-{end}",
                                "--dump-header", str(header), "--output", str(body), url], check=True)
                ranges = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", header.read_text())
                if not ranges:
                    raise ValueError("curl server ignored range")
                return body.read_bytes(), ranges[-1].strip()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _block(self, number):
        if number in self.blocks:
            self.blocks.move_to_end(number)
            return self.blocks[number]
        start = number * self.block_size
        end = min(start + self.block_size, self.size) - 1
        path = self.cache / f"{start:012d}.bin"
        if path.exists():
            data = path.read_bytes()
        else:
            # Different URLs prevent intermediate caches from reusing another
            # byte-range response. HF ignores this extra resolve query field.
            data, content_range = self._request(start, end)
            wanted = f"bytes {start}-{end}/{self.size}"
            if content_range != wanted:
                raise ValueError(f"Incorrect HTTP range: wanted {wanted}, got {content_range}")
            self.bytes_downloaded += len(data)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(data)
            temp.replace(path)
        if len(data) != end - start + 1:
            raise ValueError("Truncated HTTP range cache")
        self.blocks[number] = data
        if len(self.blocks) > 32:
            self.blocks.popitem(last=False)
        return data

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        parts = []
        while self.position < end:
            number, offset = divmod(self.position, self.block_size)
            data = self._block(number)
            count = min(end - self.position, len(data) - offset)
            parts.append(data[offset:offset + count])
            self.position += count
        return b"".join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        self.session.close()
        super().close()


def retained_action_indices(actions, threshold=1e-4):
    """OpenVLA regeneration rule: compare gripper to the last retained action.

    Removed controls are never stepped in the regenerated simulator timeline.
    A zero arm command that changes the gripper must remain in the sequence.
    """
    keep, previous = [], None
    for i, action in enumerate(np.asarray(actions)):
        noop = np.linalg.norm(action[:-1]) < threshold
        noop &= previous is None or action[-1] == previous[-1]
        if not noop:
            keep.append(i)
            previous = action
    return np.asarray(keep, dtype=np.int64)


def match_action_indices(source, target, atol=2e-6):
    """Match a filtered sequence to a unique ordered subsequence of a demo.

    Return None on a mismatch. Ambiguous repeated retained actions are rejected:
    a caller must never silently invent frame provenance.
    """
    source, target = np.asarray(source), np.asarray(target)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        return None
    matches = np.all(np.abs(target[:, None, :] - source[None, :, :]) <= atol, axis=-1)
    earliest = []
    cursor = 0
    for row in matches:
        candidates = np.flatnonzero(row[cursor:]) + cursor
        if not len(candidates):
            return None
        cursor = int(candidates[0])
        earliest.append(cursor)
        cursor += 1
    latest = []
    cursor = len(source)
    for row in matches[::-1]:
        candidates = np.flatnonzero(row[:cursor])
        if not len(candidates):
            return None
        cursor = int(candidates[-1])
        latest.append(cursor)
    if earliest != latest[::-1]:
        raise ValueError("Ambiguous action subsequence; use joint-state correspondence")
    return np.asarray(earliest, dtype=np.int64)


def match_demo(group, actions, joints=None, atol=2e-6):
    source = group["actions"][:]
    if joints is None or "obs/joint_states" not in group:
        return match_action_indices(source, actions, atol)
    source_joints = group["obs/joint_states"][:]
    # Combining the two observations disambiguates repeated control commands.
    return match_action_indices(
        np.concatenate([source, source_joints], axis=-1),
        np.concatenate([actions, joints], axis=-1), atol,
    )


def recover_episode(dataset, episode, source, cache_dir, demo=None):
    """Write a compact provenance-bearing state snapshot, without RGB copies."""
    import h5py
    import pyarrow.parquet as pq

    dataset = Path(dataset)
    destination = Path(cache_dir) / f"{dataset.name}_episode_{episode:06d}.hdf5"
    parquet = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    if destination.exists():
        with h5py.File(destination, "r") as f:
            if f.attrs["parquet_sha256"] != digest or f.attrs["source"] != source:
                raise ValueError("Cached state provenance differs")
        return destination
    table = pq.read_table(parquet).to_pydict()
    actions = np.asarray(table["action"], dtype=np.float32)
    joints = np.asarray(table["observation.states.joint_state"], dtype=np.float32)
    remote = HTTPRangeReader(source, Path(cache_dir) / "http") if source.startswith("https://") else None
    try:
        with h5py.File(remote or source, "r") as f:
            print("Source data attributes:", dict(f["data"].attrs), flush=True)
            selected = None
            keys = [demo] if demo else sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
            for key in keys:
                group = f["data"][key]
                raw_actions = group["actions"][:]
                converted = raw_actions.copy()
                # LeRobot release: open=1, close=0. Simulator: open=-1, close=1.
                converted[:, -1] = (1 - converted[:, -1]) / 2
                indices = retained_action_indices(raw_actions)
                if (len(indices) != len(actions)
                        or not np.allclose(converted[indices], actions, atol=2e-6, rtol=0)):
                    indices = None
                print(f"Match {key}: source={len(group['actions'])} retained={len(actions)} matched={indices is not None}", flush=True)
                if indices is not None:
                    if selected is not None:
                        raise ValueError("Multiple source demos match the same episode")
                    selected = (key, indices)
            if selected is None:
                raise ValueError("No original demo matches LeRobot actions under canonical no-op filtering")
            key, indices = selected
            group = f["data"][key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with h5py.File(temporary, "w") as out:
                for k, v in dict(source=source, demo=key, parquet_sha256=digest,
                                 episode=episode, schema_version=1).items():
                    out.attrs[k] = v
                for k in ("model_file",):
                    out.attrs[k] = group.attrs[k]
                for k, v in f["data"].attrs.items():
                    out.attrs[f"source_{k}"] = v
                out.create_dataset("states", data=group["states"][:][indices])
                out.create_dataset("original_initial_state", data=group["states"][0])
                out.create_dataset("original_actions", data=group["actions"][:])
                out.create_dataset("source_indices", data=indices)
                out.create_dataset("actions", data=actions)
                out.create_dataset("joints", data=joints)
                for key in group.get("obs", {}):
                    if "image" not in key and "depth" not in key:
                        out.create_dataset(f"obs/{key}", data=group[f"obs/{key}"][:][indices])
            temporary.replace(destination)
            if remote:
                print(f"HTTP transferred {remote.bytes_downloaded / 1e6:.2f} MB of {remote.size / 1e6:.2f} MB source", flush=True)
    finally:
        if remote:
            remote.close()
    return destination


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--source", required=True, help="Original LIBERO task .hdf5 path or pinned HF URL")
    p.add_argument("--cache-dir", default="data/libero_motion_sources")
    p.add_argument("--demo", help="Optional original demo name; actions are still checked")
    a = p.parse_args()
    print(recover_episode(a.dataset, a.episode, a.source, a.cache_dir, a.demo))


if __name__ == "__main__":
    main()
