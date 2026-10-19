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
| GPUs / microbatch / accumulation | 16 / 16 / 1 | same | same |
| Global batch | 256 | 256 | 256 |
| AdamW / LR / weight decay | (0.9,0.95) / 1e-4 / 0.01 | same | same |
| Schedule | 500 warmup, cosine to 0.01 ratio, 30000 updates | same | same |
| Precision / clipping | bf16 / 1.0 | same | same |
| Distributed / workers | ZeRO1 / 4 per GPU | same | same |
| Save / diagnostics | every5000 / every200 updates | same | same |
| W&B | offline | offline | offline |
| Action NFE at inference | 10 | 1 | 1 |

Seed42 is now set **before** model construction, identically across ranks; training then uses the existing seed+rank streams. Earlier runs seeded only inside the trainer, after constructing random projections/proprio. Consequently those historical checkpoints cannot be claimed to have bit-identical random initialization. For a strict new experiment, rerun the baseline with `scripts/libero/train_16h200.sh`; its shared parameters match the two new methods at initialization.

Method-specific differences are intentional and explicit, not hidden optimizer/data changes. The original FM loss also has its scheduler timestep weight; the two new objectives use uniform MSE as described below:

- **MeanFlow:** the original paper's Eq.(9),(11), `z=(1-t)*action+t*noise`, `v=noise-action`, target `stop_gradient(v-(t-r)*JVP(u,(v,0,1)))`. Hold condition and `r` fixed in the derivative. JVP uses forward AD, not finite differences or second-order backpropagation. Two shift5 uniform times are sorted, with 50% diagonal samples. Use uniform squared error with the baseline's per-sample padding reduction; no teacher, distillation, CFG, adaptive loss weighting, or geometry. Add a zero-output endpoint-time embedding; shared parameters are untouched. Inference is `noise-u(noise,0,1)`.
- **Direct Action MSE:** fixed zero action-query tokens and timestep1000; predict the clean normalized action directly. MSE uses the same padding mask and per-sample reduction as baseline. Inference returns that prediction directly, without subtracting it from noise or applying an ODE update. No extra endpoint embedding.

This is the [MeanFlow paper's core objective](https://arxiv.org/html/2505.13447v1), adapted to the shared VPP2 LIBERO recipe, rather than a claim to reproduce all ImageNet hyperparameters. Training compute per update is not equal: MeanFlow adds a JVP. Compare task success / action errors, not raw losses across different objectives.

The older `configs/meanflow/libero_mf.yaml` remains an **Action30k continuation** experiment. Do not use it for this fresh-init comparison.

## Production: Qizhi 2 nodes × 8 H200

Configure a PyTorch distributed job with two 8-H200 instances. Run this same command on both roles; PET variables determine rank and rendezvous. No hand-written IP or per-rank command is needed.

MeanFlow:

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-meanflow-16gpu-b16-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_meanflow_16h200.sh
'
```

Direct Action MSE:

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-direct-action-16gpu-b16-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_direct_action_16h200.sh
'
```

`activate_env.sh` sources the local Miniconda `etc/profile.d/conda.sh`, activates the absolute `miniconda/envs/vpp2` environment, and sets CUDA12.8 and PYTHONPATH. Launchers resolve their own repository root, including when invoked from another directory. They use `exec`, per-node locks, absolute checkpoint paths and an explicit training module. Experiment launchers live only in `scripts/libero/`; temporary benchmark commands belong under ignored `runs/`.

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
