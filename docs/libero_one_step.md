# LIBERO controlled one-step experiments

All new LIBERO Action runs use first-frame decoding, uint8 decoding without a float/uint8 round trip, and **4 workers per GPU**. The logical video length remains 17 frames (5 latent frames); frozen Video conditioning and action horizon do not change. Video pretraining still requires its full video.

## Shared controls

| Setting | FM baseline | MeanFlow | Direct Action MSE |
|---|---|---|---|
| Action backbone | 40-layer hidden512 Action1B | same | same |
| Shared initialization | `weights/libero/action_init.pt` plus seed42 projections/proprio | same | same |
| Frozen Video | `weights/libero/video_step010000.pt` | same | same |
| Trainable modules | Action + proprio | same, plus endpoint embedding | same |
| Data | four-suite natural mixture, same stats/context128 | same | same |
| Input / output | two cameras 224x448, action32x7, proprio8 | same | same |
| GPUs / microbatch / accumulation | auto / 256 ÷ GPUs / 1 | same | same |
| Global batch | 256 | 256 | 256 |
| AdamW / LR / weight decay | (0.9,0.95) / 1e-4 / 0.01 | same | same |
| Schedule | 500 warmup, cosine to 0.01 ratio, 30000 updates | same | same |
| Precision / clipping | bf16 / 1.0 | same | same |
| Distributed / workers | ZeRO1 / 4 per GPU | same | same |
| Save / diagnostics | every5000 / every200 updates | same | same |
| W&B | offline | offline | offline |
| Action NFE at inference | 10 | 1 | 1 |

Seed42 is now set **before** model construction, identically across ranks; training then uses the existing seed+rank streams. Earlier runs seeded only inside the trainer, after constructing random projections/proprio. Consequently those historical checkpoints cannot be claimed to have bit-identical random initialization. For a strict new experiment, rerun the baseline with `scripts/libero/train_action.sh`; its shared parameters match the two new methods at initialization.

Method-specific differences are intentional and explicit, not hidden optimizer/data changes. The original FM loss also has its scheduler timestep weight; the two new objectives use uniform MSE as described below:

- **MeanFlow:** the original paper's Eq.(9),(11), `z=(1-t)*action+t*noise`, `v=noise-action`, target `stop_gradient(v-(t-r)*JVP(u,(v,0,1)))`. Hold condition and `r` fixed in the derivative. JVP uses forward AD, not finite differences or second-order backpropagation. Two shift5 uniform times are sorted, with 50% diagonal samples. Use uniform squared error with the baseline's per-sample padding reduction; no teacher, distillation, CFG, adaptive loss weighting, or geometry. Add a zero-output endpoint-time embedding; shared parameters are untouched. Inference is `noise-u(noise,0,1)`.
- **Direct Action MSE:** fixed zero action-query tokens and timestep1000; predict the clean normalized action directly. MSE uses the same padding mask and per-sample reduction as baseline. Inference returns that prediction directly, without subtracting it from noise or applying an ODE update. No extra endpoint embedding.

This is the [MeanFlow paper's core objective](https://arxiv.org/html/2505.13447v1), adapted to the shared VPP2 LIBERO recipe, rather than a claim to reproduce all ImageNet hyperparameters. Training compute per update is not equal: MeanFlow adds a JVP. Compare task success / action errors, not raw losses across different objectives.

The older `configs/meanflow/libero_mf.yaml` remains an **Action30k continuation** experiment. Do not use it for this fresh-init comparison.

## Production: automatic GPU count (example: Qizhi 2 × 8 H200)

Global batch is fixed at 256 and accumulation at 1. The launcher reads PET_* allocation variables or counts visible GPUs locally; batch per GPU is 256 / total GPUs. Counts that cannot divide 256 fail early.

Configure a PyTorch distributed job with two 8-H200 instances. Run this same command on both roles; PET variables determine rank and rendezvous. No hand-written IP or per-rank command is needed.

MeanFlow:

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-meanflow-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_meanflow.sh
'
```

Direct Action MSE:

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-direct-action-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_direct_action.sh
'
```

`activate_env.sh` sources the local Miniconda `etc/profile.d/conda.sh`, activates the absolute `miniconda/envs/vpp2` environment, and sets CUDA12.8 and PYTHONPATH. Launchers resolve their own repository root, including when invoked from another directory. They use `exec`, per-node locks, absolute checkpoint paths and an explicit training module. Production launchers live in `scripts/libero/`; archived benchmark launchers live in `scripts/libero_debug/`, and run-specific probes/logs belong under ignored `runs/`.

Files:

- Baseline config: `configs/libero/train_action.yaml`.
- New configs: `configs/libero/train_meanflow.yaml`, `configs/libero/train_direct_action.yaml`; inherit the baseline.
- Logs: `runs/libero/${RUN_NAME}_logs/node0.log` and `node1.log`.
- Config/checkpoints/offline W&B: `runs/libero/${RUN_NAME}/`.
- Restore full optimizer/scheduler/RNG state by passing `resume=/absolute/path/to/checkpoints/state/step_005000` and using the original RUN_NAME. Verify the actual `checkpoints/state` directory in the run before resuming. Never pass release Action30k as initialization for these experiments.

Use `--dry-run` on either production script to print its resolved global batch, objective, sampler and initialization paths without allocating CUDA. Real-data GPU validation and test results are recorded separately in `docs/validation/libero_one_step.json`.

## Evaluation after training

The checkpoint records `action_sampler` and its objective. Loading it restores the appropriate one-step inference path. Training diagnostics also use one NFE, and report action L1 plus the method-specific loss.

For a full standard LIBERO evaluation, use the simulation environment and the **saved resolved config** (not an inheriting training preset):

```bash
# Run from the repository root; set this to the actual completed run.
RUN_NAME=libero-meanflow-16gpu-b16-YOUR_JOB-round0
METHOD=meanflow  # direct_action for the regression checkpoint
export PYTHON_BIN="$(dirname "$PWD")/miniconda/envs/robodojo/bin/python"
bash scripts/libero/eval.sh --config configs/libero/eval_standard.yaml --gpus 0 \
  "training_config=$PWD/runs/libero/$RUN_NAME/config.yaml" \
  "action_checkpoint=$PWD/runs/libero/$RUN_NAME/checkpoints/weights/step_030000.pt" \
  "video_checkpoint=$PWD/weights/libero/video_step010000.pt" \
  "action_sampler=$METHOD" action_nfe=1 source_action_step=0 action_step=30000 \
  "output_dir=$PWD/runs/libero/eval/$RUN_NAME"
```

This invokes the 2000-episode standard protocol; short training diagnostics are not a substitute for that success-rate measurement. No full task success rate is claimed by the startup validation.

## Validation record (2026-10-19 UTC)

- **40 regression tests passed**, including both 16-card launchers at rank0/rank1 from an unrelated working directory, shared hyperparameters, frozen Video / live Action+proprio gradients, padding reduction, compact checkpoint reload, and exactly one pass through all 40 Action blocks.
- **Real data equivalence passed** at nine indices across all four suites and boundaries: full decode versus first-frame uint8 processing, with exact first-frame pixels and identical action/proprio/context/masks apart from the documented video time-axis truncation.
- **Both methods passed on 8 H200**, full-size model, real LIBERO data, b16/GPU, accumulation2, global256, ZeRO1. All eight ranks completed four updates; step2/4 diagnostics and both compact/full-state saves succeeded. Both then restored optimizer, scheduler, sampler position and RNG from step4 and completed step5 plus one-step action diagnostics. This is a startup/resume check, not a convergence or task-success experiment.
- **Actual 16-H200 cross-node validation is pending allocation**, priority10 job `job-9ec7be7f-c87b-4265-b00d-a4cb94610020`. Platform status at handoff: resource shortage / queuing. The queued command uses the actual production launchers with b16, accumulation1, global256 and two diagnostic updates per method. No 16-card training result is claimed yet.

Machine-readable results and exact local evidence paths: [`validation/libero_one_step.json`](validation/libero_one_step.json). The existing 8-H200 notebook has been retained.

## JVP precision diagnostic (2026-10-20 UTC)

The fixed-input comparison ran successfully on GPU1 of the `libero-pro` two-H200 notebook, using the formal MeanFlow run's `step_005000.pt`. This checkpoint was already unstable during training. Two real samples (dataset indices 0 and 138856), seed42, fixed action/video noise and one detached BF16 frozen-Video cache were reused across every variant. Action shape was `[2,32,7]`, with 1960 cached video tokens. No optimizer updates or production configuration changes were made.

Three variants were tested: BF16 Action/JVP; BF16 primary prediction with only the Action JVP evaluated in FP32; and FP32 Action primary prediction plus FP32 JVP. All primary comparisons use MATH attention to avoid confounding backend and precision. JVP uses the existing derivative execution mode and implementation, including disabled checkpointing. FP32 casts cover Action parameters, action inputs, context and cached Video K/V; masks remain Boolean. Autocast and TF32 are disabled. All FP32 Linear inputs/parameters were checked at runtime (3272 checks during the JVP comparisons). We promote the same BF16 execution weights, rather than introducing different master weights. The JVP-only variant verifies that its BF16 primary predictions are bit-identical to the BF16 control.

| t pair | BF16 JVP relative L2 error vs FP32 | BF16 loss | FP32-JVP-only loss | Full FP32 Action loss | BF16 / full FP32 preclip Action gradient norm |
|---|---:|---:|---:|---:|---:|
| 0.20, 0.30 | 6.54% | 349.01 | 347.29 | 347.13 | 958 / 952 |
| 0.55, 0.65 | 8.84% | 262.72 | 264.86 | 264.53 | 1440 / 1456 |
| 0.90, 0.95 | 14.50% | 5757.76 | 6095.19 | 6095.93 | 7580 / 7764 |
| 0.99, 0.999 | 39.94% | 3253.76 | 4549.33 | 4550.30 | 9324 / 9805 |

The corresponding r pairs are `(0.05,0.10)`, `(0.20,0.30)`, `(0.30,0.40)` and `(0.40,0.50)`. These are deliberately selected non-diagonal stress cases, not a random estimate of mean training error. Gradient comparisons cover all Action parameters with the condition/proprio token held fixed; they exclude proprio backpropagation and apply no clipping. Changing only the JVP changes those gradients by 1.63%, 2.90%, 3.89% and 9.01% relative L2, with cosine similarities of 0.99987, 0.99969, 0.99956 and 0.99680.

Central finite differences in FP32 perturb `(z,t)` along `(v,1)`, hold r/conditioning fixed, and sweep h=`3e-4,1e-4,3e-5`. At h=`3e-5`, FP32 JVP relative L2 discrepancies are 0.066%, 0.101%, 0.286% and 0.717%; all cosine similarities exceed 0.99998. The convergence of this check supports the local JVP implementation, while BF16 errors become material near t=1. FP32 does not remove the large derivatives, targets, losses or gradients at this checkpoint. This experiment therefore supports FP32 JVP as a numerical-accuracy ablation, but does not establish BF16 as the cause of divergence or prove that full FP32 training is required. Training from an already unstable checkpoint cannot establish whether earlier BF16 errors contributed to reaching that state.

In this batch2 diagnostic, process PyTorch peaks were 46.85 GiB for BF16/JVP-only FP32 and 51.68 GiB for full FP32 Action. Both copies of the fixed condition cache are retained in every mode, and there is no optimizer/ZeRO; these are not production batch16 training memory measurements. After the first-case warmup, measured JVP calls including parameter casting took about 0.50–0.51 s (BF16), 0.48–0.52 s (FP32-JVP-only) and 0.47 s (full FP32 Action). This small MATH-attention probe is not a distributed throughput benchmark. Model/data/cache setup took 293.58 s and the full diagnostic 328.25 s.

Reproduce from the repository root, using a fresh output directory and an idle GPU:

```bash
source scripts/libero/activate_pro_env.sh
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN=runs/libero/libero-meanflow-job-3c26a4b4-f3e8-4f67-9d9b-347cd4bca37b-round0
"$PYTHON_BIN" -u scripts/libero_debug/compare_jvp_precision.py \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/weights/step_005000.pt" \
  --output-dir runs/libero_debug/jvp-precision-repeat
```

Raw batch/noise, logs and report are under ignored `runs/libero_debug/jvp-precision-20261020/`. The shareable numerical report, including finite-difference sweeps and the script SHA-256, is [`validation/libero_jvp_precision.json`](validation/libero_jvp_precision.json).
