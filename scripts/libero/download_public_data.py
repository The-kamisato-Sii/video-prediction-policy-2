"""Fetch pinned public MuJoCo 3.3.2 LIBERO archives for the labelled debug run.

Inventory matches VPP2's documented counts. Byte identity with the authors'
unreleased training corpus has not been established. Uses only requests + stdlib.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tarfile
import time

import requests

REPO = "yuanty/LIBERO-fastwam"
REVISION = "ee018b997c430bb12b5bf3c892d744798c5a2f91"
ARCHIVES = {
    "10": (1533534552, "eb5af5f46d9348f8ba0146cf90de718ce102f261348983c0fd2df08d50dd1059"),
    "goal": (837172899, "a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea"),
    "object": (1352739510, "732405d5deaf3019c244bacd3f26c9617f25c871750586c9968747b6cfefedda"),
    "spatial": (968794594, "7c374be75133e1a6779a0e07b9452c16294777be577b89553f1cf5350eafd8f9"),
}
CHUNK = 32 * 1024**2


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def fetch_part(cache, name, size, index):
    start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
    part = cache / f"{name}.{index:04d}.part"
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have == end - start + 1:
            return part
        if have > end - start + 1:
            raise ValueError(f"Oversized partial download: {part}")
        try:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            with requests.get(
                url, params={"vpp2_part": index, "offset": have},
                headers={"Range": f"bytes={start + have}-{end}"},
                stream=True, timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                expected = f"bytes {start + have}-{end}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor the requested byte range")
                with part.open("ab") as f:
                    for block in response.iter_content(1024**2):
                        f.write(block)
            if part.stat().st_size != end - start + 1:
                raise ValueError("Incomplete byte range")
            return part
        except (requests.RequestException, ValueError):
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/libero"))
    parser.add_argument("--cache", type=Path, default=Path("data/libero_archives"))
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    parts = args.cache / "parts"
    parts.mkdir(exist_ok=True)
    inventory = []
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {}
        for suite, (size, sha) in ARCHIVES.items():
            name = f"libero_{suite}_no_noops_lerobot.tar.gz"
            target = args.cache / name
            if target.exists():
                if target.stat().st_size != size or checksum(target) != sha:
                    raise ValueError(f"Existing archive failed SHA256: {target}")
                continue
            for index in range((size + CHUNK - 1) // CHUNK):
                pending[pool.submit(fetch_part, parts, name, size, index)] = name
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            print(f"{count}/{len(pending)} ranges verified", flush=True)
    for suite, (size, sha) in ARCHIVES.items():
        name = f"libero_{suite}_no_noops_lerobot.tar.gz"
        target = args.cache / name
        if not target.exists():
            temporary = target.with_suffix(".assembling")
            with temporary.open("wb") as output:
                for index in range((size + CHUNK - 1) // CHUNK):
                    with (parts / f"{name}.{index:04d}.part").open("rb") as source:
                        for block in iter(lambda: source.read(8 * 1024**2), b""):
                            output.write(block)
            if temporary.stat().st_size != size or checksum(temporary) != sha:
                raise ValueError(f"Downloaded archive failed SHA256: {temporary}")
            temporary.replace(target)
        destination = args.output / name.removesuffix(".tar.gz")
        marker = args.output / f".{suite}.extracted-{sha}"
        if destination.exists() and not marker.exists():
            raise FileExistsError(f"Refusing to overwrite an unverified dataset: {destination}")
        if not marker.exists():
            with tarfile.open(target) as archive:
                for member in archive.getmembers():
                    path = (args.output / member.name).resolve()
                    if not path.is_relative_to(args.output.resolve()) or not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsafe archive entry: {member.name}")
                archive.extractall(args.output)
            if not (destination / "meta/info.json").is_file():
                raise ValueError(f"Unexpected archive layout: {target}")
            marker.write_text(sha + "\n")
        inventory.append(dict(file=name, size=size, sha256=sha))
        print(f"SHA256 verified and extracted: {name}", flush=True)
    (args.output / "download_provenance.json").write_text(json.dumps(dict(
        repo=REPO, revision=REVISION, archives=inventory,
        scope="Public debug data; byte identity with VPP2 author corpus is unverified",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
