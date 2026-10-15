# MeanFlow implementation and verification

This is an opt-in implementation of the README's stage 1–3 engineering work.
Geometry/VGGT are disabled. A short synthetic probe does **not** satisfy the
README's full-size training or closed-loop acceptance gates.

## Environment on this server

```bash
source /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/miniconda/etc/profile.d/conda.sh
conda activate vpp2
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
```

`environment-cu128.txt` uses the reference PyTorch 2.11 / torchvision 0.26
versions with CUDA 12.8, compatible with this server's driver 570.
`environment-cu128.lock.txt` records the complete installed policy environment. The original
CUDA 13.0 reference remains unchanged. Install using
`bash scripts/setup/policy_env.sh`. DeepSpeed additionally needs an actual CUDA
compiler toolkit. This server has the actual NVIDIA CUDA 12.8.93 compiler at
`/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/tools/cuda-12.8`;
activating `vpp2` sets `CUDA_HOME`. To reproduce the local toolkit installation:

```bash
python scripts/setup/cuda_toolkit.py --prefix ../tools/cuda-12.8
export CUDA_HOME="$(cd ../tools/cuda-12.8 && pwd)"
export PATH="$CUDA_HOME/bin:$PATH"
```

Do not set it to the NVIDIA driver directory or fabricate `nvcc`.

The full RoboDojo Video + Action weights alone occupy about 36.85 GB. Joint
training adds gradients, Adam state and activations; D0 also needs an independent
frozen teacher. The original joint recipe cannot simply be run on one 48 GiB GPU
with ZeRO-1/2. A single-GPU dry run is a launch/configuration check, **not a claim
that full-size joint training fits**. Use sufficiently provisioned distributed
training hardware. CPU parameter offload / ZeRO-3 are not validated by this change.

## Simulator environment

RoboDojo lives in the sibling `../RoboDojo` checkout; the separate `robodojo`
Conda environment uses Python 3.11, PyTorch 2.7 cu128 and Isaac Sim 5.1.0.
The pinned checkouts are RoboDojo `08b7ee4`, IsaacLab
`afca7b09d60d8beb9c1cb28b43066499940b969b`, cuRobo
`d17b54ce32cba095c0b000c4c58777075d11de0e`, and XPolicyLab
`bb9a0b5f5136a74503b679af830bfd0a3a837d5c`.

```bash
conda activate robodojo
ROBODOJO_ROOT="$(cd ../RoboDojo && pwd)" PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  bash scripts/setup/robodojo_env.sh
```

The installer applies `patches/isaaclab_isaacsim51_dependencies.patch` to reconcile
IsaacLab's Starlette declaration with Isaac Sim 5.1's FastAPI requirement. It does
not change simulation or task behavior. The original spring-button reset patch
is also applied, as required by the existing RoboDojo guide. Assets are pinned to
HF dataset revision `35efbc7dedfdbeeb6e95fb749bd885d73d483e41`.
Simulator installation/closed-loop validation is still being completed; the
full-policy smoke below does not require the simulator.

## Initialization

Download the matching released pairs and shared Wan encoders using the existing
[RoboDojo](robodojo.md#checkpoint-downloads) / [LIBERO](libero.md) instructions.
Prepare the original training split, statistics and text embedding caches first.
Set the same FM pair for every method. RoboDojo must use joint-100k Video, not the
history-10k initializer. Never run `init_action.sh` for MeanFlow initialization.

The presets default to:

- RoboDojo: `weights/robodojo_release/checkpoints/joint2b_s100000/{action,video}.pt`
- LIBERO: `weights/libero_release/checkpoints/libero/{action_step030000,video_step010000}.pt`

Override `meanflow.student_init.action_checkpoint` and `.video_checkpoint` as
needed. Teacher paths interpolate these values; teacher and student are loaded
independently and their SHA-256 hashes must agree. `format=full_training` instead
uses `full_checkpoint`; for D0 set the teacher's format/path to the same full
checkpoint as well. All backbone and proprio keys load strictly **before** the
new zero-output end-time branch is added.

## Commands for every method

| ID | Method | Configuration suffix | Deployment action NFE |
|---|---|---|---:|
| B0 | Existing FM | Original benchmark training YAML | 10 |
| B1 | Same FM checkpoint, one Euler step | No training; evaluation `--steps 1` | 1 |
| B2 | FM continuation | Original trainer + matching `resume_ckpt` | 10; separately evaluate 1 |
| D0 | Frozen teacher trajectory average velocity | `distill` | 1 |
| M0 | Data MeanFlow | `mf` | 1 |
| K0 | K-MF learned split | `kmf_learned` | 1 |
| K1 | K-MF fixed midpoint | `kmf_midpoint` | 1 |

All four new methods have separate RoboDojo and LIBERO YAMLs. The original YAMLs
are unchanged. For example, dry-run all methods without querying/allocating GPUs:

```bash
for benchmark in robodojo libero; do
  for method in distill mf kmf_learned kmf_midpoint; do
    bash scripts/meanflow/train.sh --benchmark "$benchmark" \
      --config "configs/meanflow/${benchmark}_${method}.yaml" --dry-run
  done
done
```

The new launcher defaults to one node / one process and never inherits the
original RoboDojo launcher's 12-node default. Presets use microbatch 1 and
accumulation 288 / 256. On distributed hardware, explicitly set node/process
counts and adjust accumulation to retain the reference global batch.

```bash
# Examples for a suitably provisioned 8-GPU node. Replace paths first.
# 1 * 8 * 36 = RoboDojo reference global batch 288.
NNODES=1 NPROC_PER_NODE=8 bash scripts/meanflow/train.sh --benchmark robodojo \
  --config configs/meanflow/robodojo_distill.yaml \
  gradient_accumulation_steps=36 output_dir=runs/robodojo/d0_seed100008 \
  paths.wan=/models/Wan2.1-I2V-14B-480P paths.data=/data/prepared_robodojo \
  paths.media=/data/robodojo

# 1 * 8 * 32 = LIBERO reference global batch 256.
NNODES=1 NPROC_PER_NODE=8 bash scripts/meanflow/train.sh --benchmark libero \
  --config configs/meanflow/libero_mf.yaml \
  gradient_accumulation_steps=32 output_dir=runs/libero/m0_seed42 \
  model.model_id=/models/Wan2.1-I2V-14B-480P
```

Change only the YAML suffix and output directory for K0/K1. For a 20-update
**diagnostic with changed batch**, add:

```text
--allow-batch-change batch_size=1 gradient_accumulation_steps=1
max_steps=20 eval_at_start=false eval_every=0 save_every=20
output_dir=runs/NEW_UNIQUE_PROBE
```

This still uses the full-size model and needs adequate memory and real data.
It does not reduce layers or silently freeze student Video. Do not report it as
a matched-budget benchmark run. Full resume uses the same config/output plus
`resume=runs/.../checkpoints/state/step_000020`; this restores optimizer, generator,
RNG and data position. Weight-only initialization starts a new optimizer budget.

B0/B2 retain their existing scripts:

```bash
# RoboDojo B0: prepare COMMON_ARGS as in docs/robodojo.md first.
NNODES=1 NPROC_PER_NODE=8 REQUIRE_RDMA=0 bash scripts/robodojo/train.sh \
  "${COMMON_ARGS[@]}" gradient_accumulation_steps=12 \
  output_dir=runs/robodojo/b0
# RoboDojo B2 needs a FULL original training checkpoint, not release action.pt.
NNODES=1 NPROC_PER_NODE=8 REQUIRE_RDMA=0 bash scripts/robodojo/train.sh \
  "${COMMON_ARGS[@]}" gradient_accumulation_steps=12 \
  resume_ckpt=/checkpoints/original_full_fm.pt fm_continuation=true \
  model.skip_dit_load_from_pretrain=true output_dir=runs/robodojo/b2
# LIBERO B0 / B2 (configure original data/encoder paths in the YAML/overrides).
bash scripts/libero/train_action.sh output_dir=runs/libero/b0
bash scripts/libero/train_action.sh resume_ckpt=/checkpoints/action_step030000.pt \
  fm_continuation=true model.skip_dit_load_from_pretrain=true \
  resume_ckpt_load_video=false output_dir=runs/libero/b2
```

For B2, `fm_continuation=true` records the original source step and SHA-256,
and exported checkpoints retain the actual additional update count. Full-state
resume retains this provenance from the run's `fm_continuation.json`; keep the
flag enabled and set `resume_ckpt=null` when using `resume=...`.

Choose the same extra optimizer-update budget for B2 and D0/M0/K0/K1. B2's
full-training initializer and D0's deployment initializer must contain identical
FM parameters; verify hashes/tensors rather than assuming this from step labels.

The finite-difference ablation is explicit:
`meanflow.derivative_backend=finite_difference meanflow.finite_difference_step=0.01`.
JVP uses math SDPA with checkpointing disabled only during the detached derivative
branch. The primary prediction retains its original checkpointing and gradients.
Directional/dispersive/CFG/Riemannian variants are not implemented or silently
combined with these objectives.

## Tests and engineering probes

```bash
python -m pip install pytest
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q checks/test_meanflow.py
PYTHONPATH=src OMP_NUM_THREADS=2 python checks/smoke_meanflow.py \
  --device cuda --steps 20 --layers 40 --output runs/NEW_ENGINEERING_PROBE
PYTHONPATH=src OMP_NUM_THREADS=2 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 checks/ddp_meanflow.py
# Exercise the actual trainer, optimizer groups, accumulation and resume:
PYTHONPATH=src python checks/trainer_meanflow.py --benchmark robodojo \
  --output runs/NEW_TRAINER_PROBE
PYTHONPATH=src python checks/trainer_meanflow.py --benchmark robodojo \
  --output runs/NEW_TRAINER_PROBE --steps 3 \
  --resume runs/NEW_TRAINER_PROBE/checkpoints/state/step_000002
```

`smoke_meanflow.py` retains 40 actual MoT blocks but uses small hidden dimensions
and synthetic already-encoded inputs. It bypasses VAE/dataset loading and cannot
establish full-size memory feasibility, real-data convergence or closed-loop SR.
[Recorded engineering results](validation/meanflow_engineering_probe.json) identify
the exact scope and environment. Actual trainer tests require CUDA development
tools whenever the installed Accelerate imports DeepSpeed.

## Export and RoboDojo inference

Export a full RoboDojo MeanFlow checkpoint using its **student update step**:

```bash
bash scripts/robodojo/export.sh \
  --checkpoint runs/robodojo/d0/checkpoints/weights/step_000020.pt \
  --config runs/robodojo/d0/config.yaml --stats /data/prepared_robodojo/dataset_stats.json \
  --step 20 --output weights/robodojo_d0_update20
```

The export includes the paired student Video, trained end-time branch, objective,
source checkpoint hashes and update count. Teacher and split generator are absent
from the deployment bundle. Source step 100000 is distinct from student update 20.

An executable full-checkpoint policy test is available without Isaac Sim:

```bash
bash scripts/robodojo/smoke_policy.sh \
  --bundle weights/robodojo_release/checkpoints/joint2b_s100000 \
  --wan weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P \
  --synthetic --steps 10 --text-encoder-device cpu \
  --output runs/new_robodojo_policy_smoke
```

Replace `--synthetic` with `--observation observation.npz` for a recorded input
containing `head`, `left`, `right` RGB arrays, EE16 `state`, and string `instruction`.
The recorded CUDA 12.8 FM-10 run produced finite `32×16` actions, with identical
outputs across three repeated seeded requests: p50 2.316 s, p95 2.385 s, peak
38.41 GiB GPU memory. Its cold model load took 397 s on this filesystem.
See [full-policy report](validation/robodojo_policy_smoke_cu128.json).

This tests the real policy's normalization/history/encoder/MoT/output path and
counts actual calls to every action block. It saves action chunks, synchronized
p50/p95 request timings and peak memory. Synthetic or recorded-observation smoke
is **not closed-loop success evaluation**. For an MF bundle use `--steps 1`.

For a memory-constrained policy host, start the policy with
`--text-encoder-device cpu`. UMT5 stays on CPU; Video/Action/VAE/CLIP use the GPU.
This changes hardware placement and latency, so compare all methods using the
same placement. It does not alter seeds, observations or action normalization.

```bash
bash scripts/robodojo/server.sh --robodojo-root /path/to/RoboDojo \
  --bundle "$PWD/weights/robodojo_d0_update20" --wan /models/Wan2.1-I2V-14B-480P \
  --expected-step 20 --steps 1 --gpu 0 --port 23000 --text-encoder-device cpu
```

Generate a separate evaluation config using `configure_eval.sh --step 20 --steps 1`
and a unique run ID. Keep the original 54 entries / 42 cells / 2100 episodes,
shift 1, seed 1, history stride 25, chunk 32 and execute 24. B0 uses step 100000
and 10 steps; B1 uses the **same FM bundle** with `--steps 1` and its own run ID.
Do not label B1 as MeanFlow. See [closed-loop evaluation](robodojo.md#5-configure-and-run-closed-loop-testing)
for simulator setup, assets, reset fixes and the final audit.

LIBERO exports retain their paired frozen Video. Use a new evaluation YAML with
`action_sampler: meanflow`, `action_nfe: 1`, `source_action_step: 30000`,
`action_step: <actual_student_update>` and the actual action checkpoint; the
loader checks source step and student update metadata instead of pretending the
student is Action-30k. Keep all Standard/OOD/PRO trial budgets and seeds unchanged.

For LIBERO B1, set `action_nfe=1` on a copy of the original evaluation YAML,
retaining `action_step=30000`. For B2, use `fm_continuation=true`,
`source_action_step=30000`, `action_step=<actual_extra_updates>` and `action_nfe=10`
(or a separate one-step comparison with `action_nfe=1`). The checkpoint must carry
B2 provenance and its frozen Video SHA-256 must match; merely changing a step
label cannot bypass these checks.
