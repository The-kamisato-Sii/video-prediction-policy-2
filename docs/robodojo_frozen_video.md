# RoboDojo frozen-Video capacity experiment

This experiment freezes the complete Video expert and trains Action2B plus its
proprio projection. It is **not** the released RoboDojo joint-training baseline.
The released baseline remains in `configs/robodojo/train_100k.yaml`.

The probe reuses the upstream `LiberoActionModel` / `LiberoTrainer` implementation
of frozen-video conditioning, with the **RoboDojo** model dimensions and inputs:
40 Video layers, 40 Action layers, 17 RGB frames at 240 × 416, 9 conditioning
frames, 32 × 16 actions, proprio dimension 16, and all 5 video latent frames
visible to Action. The first-frame-only VAE shortcut is disabled. Video K/V is
computed under `no_grad`; Action is trained with the original flow-matching loss.
It uses synthetic RGB and cached-shaped synthetic text, not real demonstrations.

## Initialization

- Video: `weights/robodojo_baseline/checkpoints/initialization/robodojo_his10k.pt`.
  Author release: `Haodong082399/VPP2`, revision
  `04824ddb3aae11f13db50984ed7d540c3b77dd9f`.
  File SHA256:
  `2d2490bd60272ec37423dabafaed4a889b2b2bf3760ac96d63d551d723ad2852`.
- Action: `weights/robodojo_baseline/init/action2b_from_his10k.pt`, created with
  `scripts/robodojo/init_action.sh` from that Video checkpoint. The loader retains
  fresh random initialization for the action-specific input/output and video-K/V
  projection tensors, as specified by the initialization payload.
- Frozen Video parameters: **16,394,878,784**.
- Trainable Action and proprio parameters: **2,028,805,136**.

The earlier joint throughput probes used the released **joint-100k deployment
pair** as a timing fixture. This experiment uses **history-10k + fresh Action**.

## Reproduce the capacity check

On 2026-10-16, eight H200 GPUs (PyTorch-visible memory 139.81 GiB each),
PyTorch 2.11.0+cu128, Accelerate 1.13.0 and DeepSpeed 0.18.5 produced:

| Batch per GPU | Accumulation | Global batch | Peak reserved GiB | Steady seconds/update |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 24 | 49.81 | 5.07 |
| 12 | 1 | 96 | 81.02 | 7.37 |
| 24 | 1 | 192 | 122.57 | 10.84 |
| 27 | 1 | 216 | 132.96 | 11.53 |
| 28 | 1 | 224 | 136.42 | 11.77 |
| 29 | 1 | 232 | OOM during forward | — |

Every successful candidate completed three updates. Batches 30, 36 and 48 also
OOMed. The maximum verified per-GPU batch is **28 for this configuration**;
**24** leaves approximately 17 GiB of PyTorch-visible memory, while 28 is close
to the limit. These are eight-rank ZeRO-2 results, not standalone single-GPU
capacity. Changing resolution, history/context, implementation or accumulation
requires another check. Global batch 192/224 is different from the reference 288.

The complete frozen Video tensor checksum was unchanged after the search, and
the Action replicas remained identical after each successful candidate.
The four eight-rank gradient-reference cases (FP32/BF16 × accumulation 1/3)
passed: maximum relative L2 error below `1e-7` for FP32 and `0.00523` for BF16.
See [machine-readable results](validation/robodojo_frozen_video_capacity.json).

Activate the `vpp2` environment and set `CUDA_HOME` to the local CUDA 12.8 toolkit.
Run on the eight-GPU H200 instance, from the repository root:

```bash
bash scripts/robodojo/profile_frozen_video.sh \
  --output runs/frozen_video_capacity_new \
  --batch-sizes 3 --search-max --steps 3
```

Each candidate runs three complete optimizer updates, with accumulation 1,
BF16, AdamW LR `1e-4`, weight decay `0.01`, clipping `1.0`, and ZeRO-2 without
CPU offload. Warmup is disabled for this diagnostic; this is not a convergence
recipe. The first update at each batch is excluded from steady-state timing.
Batch candidates reuse the optimizer state. The search doubles the batch until
forward OOM, then bisects to an integer boundary. All ranks agree on forward OOM
before entering backward. A backward OOM or any correctness failure aborts the
probe and must not be reported as a successful batch.

The probe checks finite losses and gradients, actual Action/proprio updates,
identical complete Action weights across eight ranks, absent Video gradients,
and a full Video tensor SHA256 before and after the search. Loading, checksums,
dataset decoding, evaluation, and saving checkpoints are excluded from timing.
Per-rank allocated/reserved memory and timings are written as JSON. The
`capacity_summary.json` and `PASS` files are written only after final validation.

## ZeRO-2 arithmetic check

```bash
torchrun --standalone --nproc_per_node=8 checks/zero2_gradient_reference.py \
  --output runs/zero2_reference_bf16_acc3.json \
  --precision bf16 --accumulation 3
```

The same check also supports `--precision fp32` and `--accumulation 1`.
It compares distributed gradients against plain PyTorch with explicit averaging,
including shared parameters, a frozen branch and non-reentrant checkpointing.
This is a targeted arithmetic check, not a proof of correctness of every model
or DeepSpeed execution path.

ZeRO-1 shards optimizer state; ZeRO-2 additionally shards reduced gradients.
Neither shards model parameters. Consequently the frozen Video weights remain
replicated on each GPU, and the additional gradient-memory savings from ZeRO-2
apply to the trainable Action/proprio parameters.
See the [DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/).

The installed DeepSpeed 0.18.5 also has a separate performance issue in this
workload: every parameter gradient hook calls
`count_used_parameters_in_backward` over the complete trainable parameter list.
The clean-upstream joint trace recorded 5,377,761 `aten::view_as` calls, while
`AccumulateGrad` dominated CPU time. That is evidence of excessive CPU gradient
bookkeeping, not evidence that Video arithmetic intrinsically needs 29 seconds
per microbatch. A successful gradient-reference test does not remove this
performance issue.

An isolated DeepSpeed 0.18.2 comparison passed FP32 accumulation 1 but failed
accumulation 3 under the tested pattern of calling `engine.step()` only at the
accumulation boundary (relative gradient L2 error 0.745 for the first shared
weight). It was **not adopted**, and its queued full-model timing was not run.
This is an incompatibility of the tested integration, not a claim that every
0.18.2 training loop is incorrect. The active environment remains 0.18.5.
