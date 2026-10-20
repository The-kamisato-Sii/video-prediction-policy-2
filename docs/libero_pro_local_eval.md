# Local LIBERO-PRO evaluation environment

From the repository root:

```bash
source scripts/libero/activate_pro_env.sh
```

The Conda prefix is `../miniconda/envs/vpp2-libero-pro` (Python 3.10). It has its own simulator dependencies and reuses the existing `vpp2` environment's model packages via `vpp2_model_dependencies.pth`. Do not remove that source environment. Install simulator changes only into the PRO prefix. Current pins include MuJoCo 3.3.2, BDDL 1.0.1, future 1.0.0, NumPy 1.26.4, setuptools 81.0.0, and fsspec 2025.3.0.

The H200 notebook image lacks the GLVND loader (`libEGL.so.1`, `libOpenGL.so.0`, and `libGLdispatch.so.0`) despite providing the NVIDIA vendor EGL driver. Normally install `libegl1 libopengl0` through the system package manager. This workspace also provides these user-space loaders under `../tools/libero-pro-glvnd/lib`, copied from the Ubuntu 22.04 host with package versions and copyright saved alongside them. Activation adds that directory to `LD_LIBRARY_PATH` when present; it does not replace the container's NVIDIA driver. Because this image cannot reach the Ubuntu HTTP package mirror, the shared loaders are used here. `opencv-python-headless==4.11.0.86` is installed inside the PRO prefix to avoid inheriting the training environment's desktop OpenCV and its `libGL.so.1` dependency.

`third_party/libero_pro_runtime/` holds rpent-liberopro 0.2.0, rpent-libero 0.2.0 and robosuite 1.5.2 installed with `--no-deps --target`. The worker adds this runtime to its own PYTHONPATH. Assets come from `third_party/LIBERO-PRO-assets/libero/libero/assets/`. Task and init metadata are pinned to `c86fc3b8293185a6f373677018ff3e37f8391602` under `third_party/libero_pro_metadata/`; `prepare_pro.sh` supplies these to the isolated runtime. Local Hugging Face `.cache` directories are excluded when locating metadata trees.

The local Action-30k checkpoint is:

```text
runs/libero/libero-action-16gpu-b16-job-e252bb1f-b0fd-499c-bf63-fcc4f26d16c8-round0/checkpoints/weights/step_030000.pt
```

Its batch16 × 16-GPU training layout is accepted by the checkpoint validator; the validator checks a reference global batch of 256 instead of assuming a fixed per-GPU batch32. It does not recover the historical world size from compact checkpoint metadata.

For the VPP2 public800 protocol (800 episodes), use:

```bash
source scripts/libero/activate_pro_env.sh
RUN="$PWD/runs/libero/libero-action-16gpu-b16-job-e252bb1f-b0fd-499c-bf63-fcc4f26d16c8-round0"
bash scripts/libero/eval.sh --config configs/libero/eval_pro.yaml --gpus 0 \
  "action_checkpoint=$RUN/checkpoints/weights/step_030000.pt" \
  "training_config=$RUN/config.yaml" \
  "output_dir=$PWD/runs/libero/eval/our-baseline-30k-pro"
```

Use a new output directory for a new evaluation. Add `--dry-run` to inspect the contract before launching. This protocol covers Position/Task only; it is separate from a full five-axis PRO evaluation. Single-GPU smoke artifacts live under ignored `runs/libero_pro_single_gpu/`; small diagnostic runs do not establish a formal success rate.

## Single-GPU check (2026-10-20)

The local RTX 4090 reports 49140 MiB in nvidia-smi (the CUDA error reports 47.38 GiB total). Simulator reset/camera checks passed for Spatial Position task0 and Spatial Task task0; dependency checks passed. Full Video10k + Action30k + encoders loaded. The first policy call ran out of GPU memory in frozen Video KV prefill (RoPE allocation), before completing any episode. Peak PyTorch allocated memory was 45.74 GiB. This is not a passed rollout or a success-rate result. The user stopped further attempts; no CPU offload was enabled. Logs remain in `runs/libero_pro_single_gpu/smoke.log`.

## Two-H200 timing comparison

`scripts/libero_debug/benchmark_pro.sh` runs a bounded timing diagnostic with the same Action30k/Video10k and full GPU residency. It first evaluates eight episodes on GPU0, then the identical eight episodes across GPU0/GPU1. The sample is task0 from each of the four base suites × Position/Task, using initial-state index/seed1, original episode horizons, ten denoising steps, ten executed actions per replan, and rollout video recording.

Every worker loads an independent complete policy. A file barrier starts timing only after all workers have finished model setup. The profile separates setup, model construction, checkpoint restore and rollout wall time; it records resident and peak allocated/reserved CUDA memory. NVIDIA-SMI sampling records total process-independent GPU memory/utilization. Inherited torchrun rank variables are removed for these independent workers.

Artifacts are written to `runs/libero_pro_2gpu/${TRAIN_JOB_ID}-round${RUNNING_ROUND}/`. Each phase has per-GPU logs, manifests, episode JSONs/videos and profiles. The final `summary.json` reports speedup and a rough public800 duration extrapolated from the eight-episode two-GPU sample. This small task0 sample is not a full-benchmark success rate and does not establish a precise full-run duration.

The earlier queued task `job-0d569601-f443-48ff-bb08-8e6503da4388` is stopped. The active check uses the user's `libero-pro` notebook (`2e01c813-df25-4fdb-8333-1dfe823cf8fb`), two H200 GPUs, 40 CPU and 400 GiB RAM in `具身大模型专项空间`.

To run only the parallel diagnostic, use a fresh run name:

```bash
TRAIN_JOB_ID=pro-parallel-check bash scripts/libero_debug/benchmark_pro.sh --phase two_gpu
```

`--phase single_gpu` selects the serial diagnostic; omitting `--phase` runs both and compares their speed. An isolated phase's `summary.json` contains its own measurements, without a speedup comparison. Initial notebook attempts exposed missing EGL and desktop OpenCV libraries before any model rollout; those failed directories are retained separately from subsequent runs.

### Measured two-H200 result (2026-10-20)

Run directory: `runs/libero_pro_2gpu/notebook-2e01c813-headless-20261020-round0/`. Both workers completed all four assigned episodes with `status: PASS`; eight episodes total, independent full policies, original horizons and 10-NFE inference. No CPU offload was used. Simulator preflight and `pip check` passed after the loader/headless OpenCV fixes.

| Measurement | GPU0 | GPU1 |
|---|---:|---:|
| Load/setup seconds | 931.75 | 931.52 |
| Resident PyTorch GiB | 44.50 | 44.50 |
| Rollout peak allocated GiB | 46.59 | 46.59 |
| Rollout peak reserved GiB | 46.64 | 46.64 |
| NVIDIA-SMI sampled peak MiB | 49154 | 49154 |
| Spatial episode seconds (includes first-use overhead) | 106.99 | 107.35 |
| Object episode seconds | 17.68 | 13.52 |
| Goal episode seconds | 29.05 | 29.36 |
| Long-horizon episode seconds | 52.78 | 43.82 |

The shared barrier-to-last-worker rollout wall time was 208.14 seconds. Load/setup is separate; setup plus rollout was about 19 minutes. Six episodes after the two first-use episodes averaged 31.04 seconds. A two-GPU public800 extrapolation is about 3.45 hours from that warm average, or 5.78 hours from the full eight-episode sample including first-use overhead, plus about 16 minutes of startup for this measured launch. These are timing estimates from task0 of each cell, not a full run or a formal success-rate result. The serial phase was not run successfully in this check, so no measured single-versus-two-GPU speedup is claimed.
