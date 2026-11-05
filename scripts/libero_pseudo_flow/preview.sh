#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero_pseudo_flow/activate.sh
suite=${1:?suite};gpu=${2:-0};episode=${3:-0}
out="runs/libero_pseudo_flow_4rc_20261102/previews/$suite"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="$gpu"
"$PYTHON_BIN" -m vpp2.libero_pseudo_flow.extract --suite "$suite" --checkpoint weights/4RC/model.safetensors --resolution 504 --block 16 --pair-batch 4 --episode "$episode" --max-frames 64 --preview-root "$out" --status "$out/status.json" --code-revision "$(git -C ../tools/4RC rev-parse HEAD)" --parity --preview-tracks > "$out/extract.log" 2>&1
"$PYTHON_BIN" -m vpp2.libero_pseudo_flow.visualize --episode-dir "$out/$(printf 'episode_%06d' "$episode")" --output "$out/4rc_dt4.mp4" > "$out/visualize.log" 2>&1
"$PYTHON_BIN" -m vpp2.libero_pseudo_flow.compare_gt --episode-dir "$out/$(printf 'episode_%06d' "$episode")" --output "$out/gt_vs_4rc_dt4.mp4" > "$out/compare.log" 2>&1
"$PYTHON_BIN" -m vpp2.libero_pseudo_flow.tracks --episode-dir "$out/$(printf 'episode_%06d' "$episode")" --output "$out/4rc_tracks.mp4" > "$out/tracks.log" 2>&1
