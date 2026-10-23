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

Method-specific differences are intentional and explicit, not hidden optimizer/data changes. The original FM loss also has its scheduler timestep weight; Direct Action retains uniform MSE; fresh LIBERO MeanFlow now uses the adaptive weighting described below:

- **MeanFlow:** the original paper's Eq.(9),(11), `z=(1-t)*action+t*noise`, `v=noise-action`, target `stop_gradient(v-(t-r)*JVP(u,(v,0,1)))`. Hold condition and `r` fixed in the derivative. JVP uses forward AD, not finite differences or second-order backpropagation. Two shift5 uniform times are sorted, with 50% diagonal samples. New runs use detached per-sample adaptive weights `(valid_element_MSE + 0.001)^(-0.5)`, retaining the baseline padding and batch reduction. Historical uniform runs remain uniform; use `meanflow.loss_weighting=uniform` to reproduce that loss. No teacher, distillation, CFG or geometry. Add a zero-output endpoint-time embedding; shared parameters are untouched. Inference is `noise-u(noise,0,1)`.
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
export RUN_NAME="libero-mf-adaptive-p05-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
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

<a id="oct23-ablations"></a>

## 2026-10-23: diagonal sampling and Direct Action ablations

Each preset inherits the existing production recipe above. Global batch256, accumulation1, LR1e-4, warmup500, cosine30000, save5000, BF16, frozen Video10k, fresh Action/proprio, first-frame uint8 decoding, workers4 and offline W&B are unchanged.

| Production script in `scripts/libero/` | Input / loss | Difference from its control |
|---|---|---|
| `train_meanflow_diagonal_fix.sh` | MeanFlow + adaptive p=.5,c=.001 | Only diagonal time: independent shifted-uniform instead of sorted max; diagonal probability remains .5, scheduler shift5 |
| `train_direct_action_mae.sh` | Zero input / masked MAE | Only MSE → MAE |
| `train_direct_action_learnable_mae.sh` | Learnable input / masked MAE | Versus zero-input MAE: only input becomes trainable |
| `train_direct_action_learnable_mse.sh` | Learnable input / masked MSE | Versus original Direct Action MSE: only input becomes trainable |

Configs have the same names under `configs/libero/`, with `.yaml`. `train_meanflow.sh` retains historical `sorted_max`; `train_direct_action.sh` retains zero-input MSE. No existing checkpoint is reclassified as a new experiment.

**Diagonal sampling.** Both endpoints of a diagonal row use one fresh `td=5*u/(1+4*u)`. Off-diagonal max/min sampling, JVP, detached target, adaptive weights and loss reduction are unchanged. The historical branch consumes exactly the old RNG calls; independent mode consumes one additional batch-sized draw. `interval.train_shift` in resolved metadata records the actual action scheduler shift; conflicting values fail validation. Only LIBERO data_meanflow supports independent mode. Compact and full-state resumes reject changed sampler source, probability, mode or actual shift. Missing historical mode means sorted_max; missing shift is recovered from saved resolved config, otherwise resume fails with an explanation.

**Direct regression.** MAE is the mean absolute error over valid normalized-action elements per sample, then the batch mean, preserving the existing padding and all-padded-row reduction. Learned queries are a zero-initialized `nn.Parameter` of shape `[1,32,7]` before the unchanged Action encoder: 224 additional trainable scalars. Each action position has its own query, shared across samples. Initialization preserves the fixed-zero control's input and all shared weights; training, inference and checkpoint reload use the same query parameter. This experiment does not replace the encoder with hidden-dimension query embeddings. Both MAE and MSE still predict clean normalized actions in one forward pass, at timestep1000, with no ODE update.

Submit one experiment per distributed job. On each allocated role, run this same command; choose `METHOD` from the four names in the table (without `train_` / `.sh`):

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
METHOD=meanflow_diagonal_fix
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-${METHOD}-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash "scripts/libero/train_${METHOD}.sh"
'
```

The launchers automatically use batch32 on 8 GPUs or batch16 on 16 GPUs. Use distinct run names for distinct experiments. No new formal training job is submitted by the validation procedure.

To reproduce the small-batch execution probe on an idle two-GPU notebook (this is **global4**, not the production global256 recipe):

```bash
source scripts/libero/activate_env.sh
export PYTHONPATH="$PWD/checks:$PYTHONPATH"
export NNODES=1 NPROC_PER_NODE=2 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29683
export VPP2_TRAIN_MODULE=libero_ablation_smoke VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1 OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True VPP2_SMOKE_STEPS=4
METHOD=direct_action_learnable_mae
bash scripts/train.sh --benchmark libero --config "configs/libero/train_${METHOD}.yaml" --allow-batch-change \
  "output_dir=$PWD/runs/libero_debug/${METHOD}-smoke-repeat" overwrite=false \
  batch_size=2 num_workers=4 log_every=1 eval_every=0 wandb.enabled=true wandb.mode=offline
```

The probe shortens only the loop after constructing the original 30k scheduler. It saves/reloads the compact checkpoint and runs one action inference on each rank. Query variants additionally require nonzero finite query gradients/updates and identical query parameters across ranks. It writes `smoke_rank0.json` and `smoke_rank1.json`; it intentionally avoids repeatedly serializing the unchanged frozen Video expert. Production save/resume behavior is unchanged.

New MeanFlow diagnostics use global sum/count reductions for diagonal/off-diagonal t means and five-bin counts (edges 0,.2,.4,.6,.8,1; the last bin includes 1), plus gap bins and off-diagonal valid-element correction RMS. Empty groups retain count0 and a zero-valued mean placeholder. Old raw/weighted MSE, weights and correction RMS remain available. Rank-zero-only evaluation performs no training collectives.

CPU/config validation: **79 distinct checks passed** (40 existing MeanFlow/adaptive/launcher checks and 39 new ablation checks). Coverage includes controlled RNG, empirical shifted CDFs, diagonal JVP skipping, padding/MAE gradients, query updates, compact inference reload, full optimizer/scheduler query round trips, sampler resume guards, and 8/16-GPU dry runs from unrelated directories. A cold-import timeout in the existing two-rank test was fixed by importing evaluation dependencies on both ranks before creating the process group; its original 45-second collective timeout remains unchanged, and the rerun passed.

**Full-size GPU validation: all four passed** on notebook `vpp2-one-step-ablations-2h200` (`7afff45e-8a19-40c0-85c1-e4599d9ce4b2`). Two H200, ZeRO1, BF16, real LIBERO data, batch2/GPU, accumulation1 (diagnostic global4), four updates per method; the 30k schedule remained intact. Every run saved a compact checkpoint; both ranks reloaded it and completed exactly one Action forward on a synthetic observation. Learned-query gradients and updates were finite/nonzero and the updated parameters matched exactly across ranks.

| Experiment | Final training loss (different objectives) | Peak allocated GPU memory | Both ranks |
|---|---:|---:|---|
| `meanflow_diagonal_fix` | 1.1318 | 49.54 GiB | PASS |
| `direct_action_mae` | 0.3891 | 45.71 GiB | PASS |
| `direct_action_learnable_mae` | 0.3891 | 45.71 GiB | PASS |
| `direct_action_learnable_mse` | 0.2795 | 45.71 GiB | PASS |

The loss values establish finiteness only; MAE, MSE and adaptive MeanFlow losses are not comparable quality metrics. Zero-input MAE additionally saved the full DeepSpeed state. That write included the frozen Video expert and was slow on the shared filesystem; subsequent probes used compact saves. Full optimizer/scheduler query restoration and the following update were checked separately on a small model.

Raw logs, checkpoints, rank reports and offline W&B are under `runs/libero_ablations_validation_20261023/`. Shareable evidence, source hashes and limitations: [`validation/libero_ablations_20261023.json`](validation/libero_ablations_20261023.json). The notebook remains available. This is not a 16-GPU cross-node run, a production batch16/32 memory test, or a LIBERO environment rollout. Short startup tests establish execution and checkpoint compatibility, not convergence or task success; the README's longer 1–2k and closed-loop comparisons remain future experiments.

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


### 单 H200、batch=16 完整训练步精度测试（2026-10-20）

脚本：`scripts/libero_debug/benchmark_meanflow_full_step.py`；原始结果：
`docs/validation/libero_full_step_precision_b16.json`。
在 `libero-pro` 两卡交互式建模的物理 GPU 1 上顺序测试，另一张卡未用于本次测试。
两组均从 MeanFlow step 5000 权重初始化，同样的 160 条真实样本、逐步相同随机种子，
workers=4、首帧解码、减少类型转换。冻结 Video 使用 BF16；FP32 组将 Action（主前向、JVP、反向）
及 proprio 参数改为 FP32，并转换 Action 条件缓存。TF32 关闭，主前向使用默认 SDPA dispatch，
JVP 使用 MATH。每步包含 dataloader 等待、输入编码、重新计算 Video prefill、Action/JVP、反向、
梯度裁剪、AdamW 更新；没有复用上一轮的 Video cache。

| 模式 | 完成步数 | 首步 | 后 9 步平均 | PyTorch 峰值 allocated |
|---|---:|---:|---:|---:|
| BF16 | 10/10 | 28.39 s | 3.434 s | 113.80 GiB |
| Action FP32 | 0/10 | 第一步 OOM | 无有效完整步测量 | OOM 时 138.92 GiB，非成功运行峰值 |

BF16 后 9 步平均分项：输入编码 0.488 s、Video prefill 2.000 s、Action 前向及 JVP 0.649 s、
反向 0.195 s、裁剪及 optimizer 0.101 s；数据预取后等待约 0.00013 s。
BF16 reserved 峰值 114.03 GiB。首步包含 worker 等待及 CUDA 初始化，不能代表稳定吞吐。

FP32 在 `slice_condition` 对非对角样本切片复制 Video KV 时 OOM，尚未完成第一步反向，
也尚未初始化 AdamW moments。去掉 FP32 参数不必要的额外 master 权重后重测，仍在同一位置 OOM：
卡容量 139.81 GiB，进程占用 139.65 GiB，剩余 152 MiB，额外申请 268 MiB 失败。
因此不能根据这次实验给出 FP32 的减速比例，也不能将 batch=2 的显存比例外推到 batch=16。

这是单卡计算基准，不使用 DeepSpeed/通信。两组均用新建 FP32 AdamW 状态、LR=1e-4、
betas=(0.9,0.95)、weight_decay=0.01、clip=1；BF16 有 FP32 master，FP32 直接更新参数。
未恢复正式训练的 optimizer/scheduler，因此不等价于正式任务断点续训。
BF16 十步均有限，但 checkpoint 本身已有较大的 loss 波动，十步不能证明收敛。
正式训练任务和生产训练代码未改动。

复现（空输出目录，GPU 1 空闲时）：
```bash
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_pro_env.sh
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 VPP2_TORCH_LOAD_MMAP=1 VPP2_VIDEO_DECODE_UINT8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
RUN=runs/libero/libero-meanflow-job-3c26a4b4-f3e8-4f67-9d9b-347cd4bca37b-round0
MODE=bf16 # 改为 fp32_action 测试 FP32
"$PYTHON_BIN" -u scripts/libero_debug/benchmark_meanflow_full_step.py \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/weights/step_005000.pt" \
  --mode "$MODE" --steps 10 --batch-size 16 \
  --output-dir "runs/libero_debug/full-step-retest-$MODE"
```


## Adaptive MeanFlow default (2026-10-20)

New `scripts/libero/train_meanflow.sh` runs default to `loss_weighting=adaptive`,
`adaptive_weighting.power=0.5`, `adaptive_weighting.c=0.001`. Explicit `uniform`
reproduces the historical loss. Missing fields in old resolved configurations and
checkpoint metadata mean uniform; both compact weight loading and full training-state
resume reject a silent change of mode or adaptive parameters.

The detached per-sample weighting formula agrees with the official
[JAX implementation](https://github.com/Gsunshine/meanflow/blob/main/meanflow.py#L216-L227)
and the author's [PyTorch implementation](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py#L52-L60).
There are intentional differences from the ImageNet recipe: this LIBERO experiment
uses valid-action-element **mean** error and p=0.5,c=0.001; the official JAX code uses
per-image **sum** error and defaults to p=1,c=0.01. Target and weight are detached,
residual/reduction/weight calculations use FP32, while the existing BF16 network/JVP
execution is preserved. No weight renormalization/clipping or scheduler loss weight
is introduced. The existing terminal-token mask and batch denominator are retained.

`train/mf/raw_mse`, `train/mf/weighted_mse`, diagonal/off-diagonal raw MSE, weight mean/max,
group counts and correction RMS are logged to offline W&B. These statistics use globally
summed valid-sample counts; they can differ from the training loss denominator when
fully padded samples occur. A group with count zero has no error estimate. Rank-zero-only
evaluation disables training collectives. Lower weighted loss alone is not convergence evidence.

Validation artifacts and run status are recorded in
`docs/validation/libero_adaptive_loss.json`. The test suite covers formula/gradient
agreement, p=0 equivalence, padding/all-padding, two-rank reduction with empty groups,
rank-zero-only real-model evaluation, default/invalid configurations, compact metadata,
and full-state loss compatibility guards. The separate bounded real-data test uses
`scripts/libero_debug/adaptive_smoke.py`; its scheduler remains configured for 30000
steps while only the smoke loop is limited. Formal launch uses the ordinary production
script with no diagnostic stop limit.


The combined 39-test suite and final 9-test adaptive regression suite passed
(40 unique tests, including the added full-state compatibility guard). On the two-H200
`libero-pro` notebook, real-data BF16 ZeRO-1 training with batch16 per GPU completed
updates 1–4, rank-zero evaluations at steps 2/4, full checkpoint saving, and restoration
of model/optimizer/scheduler/RNG/dataloader state followed by updates 5–6 and evaluation.
The saved/resumed loss remains adaptive p=0.5,c=0.001; both schedule and MeanFlow budget
remain 30000 steps. Step6 weighted loss=1.2307, raw MSE=1.5271, preclip norm=3.3038,
LR=1.40e-6 (warmup continuing), fixed-sample action L1=0.652843. These are short-run
engineering observations, not a convergence or closed-loop result.

The initial test workers were started before the rank-zero collective correction and
stalled at evaluation; they were stopped. The restarted test audits the loaded model/loss
APIs inside each worker and passes evaluation and full restore. Full-state disk save/read
was slow on the shared filesystem; a diagnostic stack dump confirmed Torch/DeepSpeed
checkpoint I/O, rather than a training collective hang during the successful restore.

Formal job: `vpp2-libero-meanflow-adaptive-1020`,
`job-71de1e8c-75b5-4c44-8969-26acbec0b028`, HIGH priority, two 8-H200 instances.
It starts fresh, with batch16/GPU, global256, accumulation1, BF16/ZeRO1, workers4,
LR1e-4 with 500 warmup and 30k cosine, save5000, eval200, offline W&B `VPP2-LIBERO`.
There is no diagnostic early-stop wrapper in this formal job. The 17 compared training/data
controls match the previous uniform MeanFlow resolved configuration.
Its output directory is
`runs/libero/libero-mf-adaptive-p05-job-71de1e8c-75b5-4c44-8969-26acbec0b028-round0`;
node logs use the same name with `_logs/node0.log` and `_logs/node1.log`.
