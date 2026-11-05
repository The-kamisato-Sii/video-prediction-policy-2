LIBERO flow/action shared DiT (isolated implementation)

4RC pseudo-flow extraction, pinned dependencies/weights, normalization, MotionVAE
x10/scaling_factor, tensor shapes and matched-source training commands:
  docs/libero_pseudo_flow.md
  (GitHub link: ../../docs/libero_pseudo_flow.md)

Head-only motion supervision is the default from2026-10-25.
  flow.motion_cameras=[image] supervises external/head-camera point maps + flow.
  RGB conditions still include image AND wrist_image. Use
  'flow.motion_cameras=[image,wrist_image]' only for an explicitly dual-view run.
  From2026-10-26, mu/S use only the selected label cameras over the sampled
  eight frames. For [image], wrist points and masks cannot affect mu/S.
  Point maps use (P-mu)/S; flow uses F/S before the unchanged VAE x10 factor.
  The LZ4 payload retains metric values and does not need conversion again.
  Selecting one label view does not yet avoid decompressing the other view.
  resolved_config records motion_normalization=selected_views_clip_mean_radius_v1;
  training cannot resume optimizer state across the old/new normalization.
  Old dual-view checkpoints cannot resume into head-only models.

Default task-homogeneous interleaved global batches added2026-10-25.
  See scripts/libero_flow/SAMPLING.md. Global batch256 and30000 steps unchanged.
  flow.batch_sampling=mixed retains the old sampler for comparisons/exact old-run resume.

One-pass reconstruction ablation added2026-10-23:
  See scripts/libero_flow/reconstruction/README.txt and its9 training wrappers.
  Explicit --objective direct_reconstruction selects queries, a random trainable
  decoder and masked3D reconstruction. Existing launchers default to latent FM.

The original src/vpp2/libero, models, MeanFlow, configs/libero and scripts/libero
files are imported read-only. No monkeypatching. The dependency overlay is a
separate system-site-packages venv; diffusers0.31.0 is NOT installed into the
running baseline environment.

Architecture
  Observations, frozen Wan14B video_step010000, action_init backbone, language
  cache and proprio match configs/libero/train_action.yaml.
  Indicator0=action,1=flow is a learned embedding added to shared tokens and
  timestep conditioning. Both tasks traverse the SAME40-layer512-wide ActionDiT.
  The original ActionDiT linear output head is frozen and unused.
  MotionCrafter encodes each view/frame independently (spatial8x,time1x).
  Each view combines geometry4ch+flow4ch. Default external-only labels:
  [B,1,6,8,224,224] -> [B,8,8,28,28].
  Explicit dual-view mode channel-packs to[B,16,8,28,28].
  Conv3d kernel=stride=(4,7,7)/(2,7,7)/(1,7,7), padding0 ->32/64/128 tokens.
  Action32x7 is linearly embedded and expanded by repeat_interleave1/2/4.
  Conv1d kernel=stride=1/2/4 -> SiLU ->1x1 projection yields32x7 velocity.
  Flow hidden states ->ConvTranspose3d -> full VAE latent velocity; a bounded
  noisy-latent preconditioning skip preserves noise outside the patch bottleneck.
  Ten latent FM steps -> MotionVAE decode -> one224x224 external view of8 point/flow maps.
  Returned3D maps are in the normalized target frame, not unnormalized metres.
  No raw224->28 learned stem, overlapping patches, padding or extra frames.

Loss definition
  Action: baseline normalized action FM, shifted-uniform sigma, shift5.
  Flow: FM in the official scaled joint VAE latent, same sigma scheduler.
  MotionVAE decode is used for3D output and reconstruction checks. Optional
  flow.decoded_3d_weight enables masked clean3D reconstruction through the frozen
  decoder; default0. This is NOT a dense3D velocity FM objective through a
  nonlinear VAE, and should not be described as one.
  For flow preconditioning, v=c_skip*z_sigma+c_out*head(hidden), where
  c_skip=(2*sigma-1)/(sigma^2+(1-sigma)^2), c_out=1/sqrt(sigma^2+(1-sigma)^2).
  These coefficients are fixed across all3 patch variants.

Data
  Reuse exact seed42 baseline episode manifests:214/428/856 action episodes
  out of1712. flow_episode_scope=all considers ALL1712 episodes independently
  of the action subset; only complete available flow records are included. A selected
  episode contributes separate action and flow task records. Parquet columns
  exclude action for flow samples; action samples never open motion files.
  flow_episode_scope=complementary explicitly reproduces the old partition.
  From2026-10-25, missing_motion=skip is the production default: missing or
  incomplete flow excludes ONLY the flow record. Selected action records remain
  eligible even when replay/flow generation failed. No extra action filtering.
  Missing_motion_policy, skipped_flow_episodes, action_episodes_without_flow
  and the explicit missing list are recorded in dataset_audit.json.
  flow.missing_motion=error remains available as an optional strict check.
  Flow episode/start sets are identical across action fractions, given identical
  completed GT. Uniform task-start sampling remains; the action fraction is
  label coverage, not minibatch task probability. With equal valid starts this
  would be roughly1/9,1/5,1/3 with full flow coverage; missing flow and retained
  action tails change the exact ratios.
  dataset_audit.json records the actual sampled_action_fraction and overlap.
  New default run names include allflow_ to separate this data-policy change.
  Do not resume old complementary sampler/optimizer state into all-flow runs.
  Baseline action starts and padded tails are retained and masks preserved.
  Flow requires start+32<episode_length for BOTH dt1 anddt4: sources0,4,...,28;
  endpoints1,5,...,29 or4,8,...,32. All8 valid flow fields are retained.
  Therefore episode-label fraction, available-start fraction and sampled task
  fraction are different quantities; dataset_audit.json records the first two.
  Uniform available-start sampling matches baseline frame sampling convention;
  do not claim each minibatch contains an exact1/8 ratio. Action exposure also
  differs from an action-only run at equal total steps/global batch.
  Both production and diagnostic launchers freeze one available-flow manifest
  before launching ranks and reuse it on resume, avoiding races with generation.
  Missing flow is filtered when building the dataset, not by dropping batches
  inside workers. Global batch256 therefore remains unchanged.
  Invalid/corrupt payload read errors are not silently swallowed.
  Validation: runs/libero_flow_implementation/skip_missing_flow_20261025/.

Launch examples (from repository root)
  bash scripts/libero_flow/setup.sh
  bash scripts/libero_flow/train_32tokens_1of8.sh --dry-run
  FLOW_DT=1 bash scripts/libero_flow/train_64tokens_1of4.sh
  FLOW_DT=4 bash scripts/libero_flow/train_128tokens_1of2.sh
  Every32/64/128 x1of8/1of4/1of2 combination has a wrapper. FLOW_DT defaults4.
  Each invocation starts ONE training run; there is no automatic full sweep.
  Add flow.action_only=true for architecture-matched action-only controls.
  Use a new RUN_NAME for a fresh experiment; existing output is not overwritten.

Controlled recipe
  30000 optimizer steps; effective global batch256; AdamW lr1e-4,
  betas0.9/0.95, weight_decay0.01, warmup500, cosine floor0.01, clip1,
  seed42, BF16, ZeRO1, save5000, same baseline initialization/observations.
  Per-GPU effective batch=256/(nodes*visible GPUs). Default microbatch cap:
  16 for>=120GiB,4 for>=75GiB,1 otherwise. The remaining factor is gradient
  accumulation, so optimizer steps and effective batch do not change.
  Accumulation carries across epoch boundaries; the final microbatches of an
  epoch do not trigger a smaller-than256 optimizer update.
  --micro-batch may override the cap but must divide256/world_size.
  This memory-aware accumulation differs from baseline's fixed accumulation1;
  resolved configuration records the difference. Equal10 NFE is not equal FLOPs.
  The baseline trainer/optimizer/scheduler are reused through a local subclass.
  Diagnostic max_steps changes the LR scheduler horizon as well; never compare
  diagnostic metrics with a30000-step training run.

Checks
  bash scripts/libero_flow/validate_single.sh --allow-incomplete-gt
  bash scripts/libero_flow/validate_distributed.sh
  Single-GPU checks use real labels and pretrained weights, full40-layer backbone,
  both branches' backward gradients, no frozen-module gradients,10-step sampling,
  transposed-convolution/codec output shapes and codec roundtrip RMSE.
  Validation is infrastructure correctness, not evidence of task success.
  Optional bounded timing uses the existing profiler without changing the
  baseline trainer: profiling.enabled=true profiling.collect_steps=2
  profiling.log_every=1. Reports include per-rank data_wait/forward/backward;
  CUDA synchronization perturbs timing, and backward also includes ZeRO update
  and distributed waiting. Disable profiling for production measurements.
  Sampling API: model.sample_task(collated_batch, indicator=0 or1, steps=10).
  The inherited baseline infer_action entry point is deliberately rejected: it
  would use the unused old head. The new training diagnostics use sample_task.
  LIBERO rollout deployment is implemented in vpp2.libero_flow.inference/evaluate.
  It restores the training adapters and configuration; see docs/libero_pseudo_flow.md.
  To measure single-GPU capacity with two real optimizer updates for each
  all-action/all-flow/mixed batch, and stop at the first OOM:
    bash scripts/libero_flow/measure_capacity.sh --tokens 128 --batches 16 32 64 --output runs/libero_flow/capacity_new
  The default backend is ZeRO1 with one rank; FLOW_CAPACITY_BACKEND=plain selects
  ordinary AdamW. Do not report plain AdamW capacity as ZeRO1 capacity.
  Measured2026-10-23 at128 tokens/default latent FM: single H200/ZeRO1 passed
  batch16 for all three label mixtures (max101.49GiB allocated); batch32 OOM.
  The existing2-H200 notebook passed one global256 update using16x2x8,
  with rank peaks88.72/91.80GiB. Local48GB4090/plain AdamW passed batch2
  (max45.05GiB), batch4 OOM; batch1 leaves more headroom. These are tested
  powers-of-two choices, not an exhaustive search over every integer batch size.

Dependencies
  The isolated overlay pins diffusers0.31.0 and opencv-python-headless4.11.0.86.
  Headless OpenCV avoids the missing libGL.so.1 dependency in the H200 image;
  its version matches the original environment's OpenCV without modifying it.
  Local codec: weights/MotionCrafter/geometry_motion_vae/{config.json,
  diffusion_pytorch_model.safetensors}; downloaded from TencentARC/MotionCrafter.
  Codec implementation follows official commit9cb4e9679f5f34e249945544052464ef46324bc2.
  Both geometry and flow VAE parameters stay frozen. Decoder gradients to latent
  remain available for optional reconstruction loss. No simulator is needed by
  the training dataloader beyond the generated HDF5 labels.

Fast motion storage (2026-10-23)
  Canonical simulator GT remains in motions/chunk-*/episode_*.hdf5.
  The isolated flow loader can read lossless training sidecars from:
    motions_fast/chunk-*/episode_*.mrec
    meta/motions_fast/episode_*.json
  Defaults (2026-10-25): flow.motion_backend=records, flow.motion_open_files=8.
  Training requires validated LZ4 caches and never silently falls back to HDF5.
  Missing GT still skips only the flow record; missing/corrupt/stale LZ4 for
  otherwise valid GT is an explicit error, requiring cache conversion/repair.
  flow.prefetch_factor controls queued batches per worker (default1), and
  num_workers controls workers per rank. Both are configurable in launch overrides.
  auto uses current sidecars and falls back to canonical HDF5 for missing,
  stale or corrupt caches (corruption/staleness is warned). hdf5 forces the
  original reader; records requires current valid sidecars and fails otherwise.
  Action-only samples never access either motion representation. All fractions,
  sampling,32-action horizon,8 motion sources,dt1/dt4,batch256 and training
  steps are unchanged. The isolated flow worker default is12 per rank following
  the NPZ/worker experiment below; original baseline/MeanFlow defaults are intact.

  Format: VPPMOT01 outer container, JSON header, absolute uint64 record offsets,
  followed by independently checksummed standard LZ4 frames. Stores BOTH views,
  float32 metric point maps and dt1/dt4 scene flow, plus bit-packed masks.
  Float bytes are shuffled reversibly before compression; no FP16 quantization.
  Frames are phase-major:0,4,...,1,5,...,2,6,...,3,7,... . Any valid8-source
  stride4 clip is consecutive records, read by one pread for its compressed
  payload. Opening a file additionally reads its header/index and validates
  source identity/metadata. Handles are bounded and process-local.
  The frozen MotionVAE encoding and per-clip center/mean-radius normalization
  still run in the original places; no approximate or frame-only latent cache.

  Convert existing completed GT once:
    bash scripts/libero_flow/prepare_motions.sh --workers 2
  Follow ongoing generation in a separate process:
    bash scripts/libero_flow/prepare_motions.sh --workers 2 --watch
  --suite object|spatial|goal|10 restricts the suite. --source accepts explicit
  complete HDF5 paths. --force repairs specified caches without --watch.
  --scratch-dir may select local staging storage. CPU only, nice10, no GPU.
  Status/events: runs/libero_flow_motion_conversion/{status.json,events.jsonl}.
  Original generation processes are never restarted or modified.
  Converter requires a completed replay-v3 source and its completion metadata,
  verifies EVERY raw frame and normalized clips at bothdt values, rechecks
  source identity, then publishes atomically. Concurrent converters use locks.
  Partial output is never consumed. Missing/failed simulator episodes remain
  missing; this converter cannot repair or invent them.

  Reproducible correctness check on converted real files:
    python -m vpp2.libero_flow.validate_storage --source <HDF5 paths> --output <report.json>
  This checks exact targets/normalization across temporal phases, missing and
  corrupt/stale caches, source identity, missing-label guards, and fork workers.
  Benchmark artifacts/scripts: runs/libero_flow_implementation/storage_benchmark/.
  The new-inode first-access test on H200 (4 suites,2 windows each) measured
  3.137s/clip original HDF5 versus0.668s indexed shuffled LZ4; hot repeats were
  0.182s versus0.158s. This includes normalization, not GPU training. LZ4 without
  byte shuffle was faster hot (0.122s) but larger; the chosen shuffle version
  prioritizes the current random first-access bottleneck and storage footprint.
  First access is not a claim that physical disks/server caches were cold.
  Four training sidecars total about87% of the FULL original HDF5 bytes;
  archives include extra visualization/replay fields. Keeping both therefore
  adds storage. These small-sample results do not establish full-run GPU speedup.
  LZ4 frame/checksum API reference:
    https://python-lz4.readthedocs.io/en/stable/lz4.frame.html
  Measured actual CPU loader:32 samples (4A/28F),batch8,4 workers,
  first access to motion replicas with common RGB/text/parquet prewarmed:
  HDF532.022s -> records8.800s (3.64x); all collated tensor hashes identical.
  Launch the independent persistent converter in tmux:
    bash scripts/libero_flow/start_motion_cache.sh
  Session name:libero-flow-motion-cache. Status remains in the directory above.

NPZ and worker follow-up (2026-10-23)
  Detailed conclusions, scripts and measurements:
    runs/libero_flow_implementation/npz_workers/REPORT.txt
  Tested standard raw/compressed NPZ, whole-episode/1-frame/8-frame arrays,
  four suites and exact normalized target checks. Keep LZ4: raw NPZ needs2.85x
  storage; compressed NPZ saves only3.5%, and neither wins the real loader test.
  Actual CPU loader,384 identical samples48A/336F,batch16,two warmed sweeps:
    workers/rank       4       8       12
    raw NPZ           12.79   18.80   29.09 samples/s
    compressed NPZ    11.76   17.50   28.93 samples/s
    indexed LZ4       14.14   27.04   34.86 samples/s
 2H200,global256,micro16/rank,GAS8,128tokens,dt4,three steps each:
  4->12workers reduced the second step49.56->44.68s; data wait1.42/4.63s
  ->0.28/0.24s. Mean GPU utilization stayed about54% in both settings.
  Third step restarts the diagnostic dataset epoch and took34.47/35.98s;
  more workers add restart overhead. Three steps do not establish long-run speed.
  Flow default:12workers per rank. Override by appending num_workers=4 or8 to
  any flow training command on hosts with less CPU capacity. Batch256,training
  steps and label split are unaffected. The2-worker CPU conversion daemon is
  separate from training workers and remains unchanged.

LZ4/prefetch debug on vpp2-one-step-ablations-2h200 (2026-10-25)
  Detailed report: runs/libero_flow_implementation/prefetch_20261025/REPORT.txt
  Sixteen validated episodes; actual all-flow batch16; two32batch runs/setting.
  Mean batch time including startup: workers4/8/12/16 withprefetch1:
  1.326/0.757/0.647/0.594s. Workers12 withprefetch2/4:0.522/0.531s.
  All6272 flow samples used strict LZ4; all1589 production caches passed a
  source-signature/metadata/size audit (not a new full payload scan).
  Direct-reconstruction128tokens,2H200,global256,three updates per setting:
  compare updates2-3, prefetch1/4:88.587/88.810s per update, data wait0.071/0.065%.
  GPU0/GPU1 utilization65.16/85.75% versus67.43/84.61%; no material training gain.
  Default remains12workers/prefetch1. Use flow.prefetch_factor=2 for more input
  buffering if useful;4 adds memory without measured training benefit here.
  These are short subset diagnostics, not a full-dataset sustained-speed claim.
