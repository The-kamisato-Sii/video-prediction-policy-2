#!/usr/bin/env bash
# Activate the separate Python 3.11 simulator environment before running.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sim_root="${ROBODOJO_ROOT:?Set ROBODOJO_ROOT to the pinned simulator checkout}"
sim_root="$(cd "$sim_root" && pwd)"
python_bin="${PYTHON_BIN:-python}"
constraints="$repo_dir/environment-robodojo-cu128.txt"
for name in robodojo_optional_kit_args robodojo_request_timeout; do
  runtime_patch="$repo_dir/patches/$name.patch"
  if git -C "$sim_root" apply --check "$runtime_patch" 2>/dev/null; then
    git -C "$sim_root" apply "$runtime_patch"
  else
    git -C "$sim_root" apply --reverse --check "$runtime_patch"
  fi
done
lab="$sim_root/third_party/IsaacLab"
patch="$repo_dir/patches/isaaclab_isaacsim51_dependencies.patch"
if git -C "$lab" apply --check "$patch" 2>/dev/null; then
  git -C "$lab" apply "$patch"
else
  git -C "$lab" apply --reverse --check "$patch"
fi
"$python_bin" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com -c "$constraints"
"$python_bin" -m pip install -r "$sim_root/scripts/requirements.txt" \
  opencv-python-headless pillow matplotlib scikit-learn -c "$constraints"
"$python_bin" -m pip install -e "$lab/source/isaaclab" \
  -e "$lab/source/isaaclab_assets" -e "$lab/source/isaaclab_tasks" \
  -e "$lab/source/isaaclab_rl" -c "$constraints"
"$python_bin" -m pip install setuptools_scm wheel -c "$constraints"
"$python_bin" -m pip install -e "$sim_root/third_party/curobo[cu12]" \
  --no-build-isolation -c "$constraints"
"$python_bin" -m pip check
if [[ -d "$sim_root/Assets/Robots" ]]; then
  (cd "$sim_root" && "$python_bin" utils/update_embodiment_config_path.py)
fi
