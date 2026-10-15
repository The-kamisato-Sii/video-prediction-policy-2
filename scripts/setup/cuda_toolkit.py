"""Install NVIDIA's pinned minimal CUDA 12.8.1 compiler/runtime/headers locally."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prefix", required=True)
a = p.parse_args()
root = Path(a.prefix).resolve()
root.mkdir(parents=True, exist_ok=True)
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    manifest = directory / "redistrib.json"
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", base + "redistrib_12.8.1.json", "-o", str(manifest)],
        check=True,
    )
    data = json.loads(manifest.read_text())
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
        item = data[name]["linux-x86_64"]
        archive = directory / (name + ".tar.xz")
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", base + item["relative_path"], "-o", str(archive)],
            check=True,
        )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        subprocess.run(
            ["tar", "-xf", str(archive), "--strip-components=1", "-C", str(root)], check=True
        )
    (root / "redistrib_12.8.1.json").write_text(manifest.read_text())
if not (root / "lib64").exists():
    (root / "lib64").symlink_to("lib")
subprocess.run([str(root / "bin/nvcc"), "--version"], check=True)
print(f"export CUDA_HOME={root}")
