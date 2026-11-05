# LIBERO replay → dense 3D motion

This pipeline adds motion sidecars to existing LeRobot v2.1 episodes. The existing
RGB, parquet, normalization statistics, subset manifests and action training
loader retain their existing contracts.

Representation and visualization references:

- [MotionCrafter, §3.1 and Appendix A](https://arxiv.org/html/2602.08961v2).
- [MotionCrafter visualization code](https://github.com/TencentARC/MotionCrafter/blob/main/visualize/visualize.py): RGB-colored point clouds and colored displacement segments.
- [LIBERO demonstration regeneration](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py).

## Representation

For a pixel in frame `i`, `point_map[i,u,v]` gives the visible material point's
3D position. `scene_flow[i,u,v]` gives that **same material point's** displacement
to frame `i+dt`. It is not the subtraction of point maps at the same pixel.

Both views use the initial **agentview OpenCV camera** as one fixed reference
frame. This extends the paper's monocular convention to the two synchronized
LIBERO cameras. The camera poses are retained separately. In particular, wrist
camera ego-motion does not create nonzero flow on a static table or floor.

Depth and MuJoCo geom IDs identify each visible point and its rigid body.
For that body, the exact correspondence is:

```text
q_local = R_body(i)^T (X_world(i) - t_body(i))
X_world(i+dt) = R_body(i+dt) q_local + t_body(i+dt)
V_ref = R_reference^T (X_world(i+dt) - X_world(i))
```

Each articulated robot link is a separate rigid body. This method covers LIBERO's
rigid objects and articulated robot/fixtures; it does not model deformable meshes.

The default target grid is 224×224 per camera. Rendering and RGB verification use
the original 512×512 resolution. Depth/segmentation/point maps are sampled at
explicit native pixel centers, without blending foreground and background 3D
coordinates. `sample_pixel_x/y` describe that sampling exactly.

Two flow intervals are retained at **every** dataset frame:

- `dt=1`: one retained action interval, nominally 0.05 seconds.
- `dt=4`: four retained action intervals, nominally 0.2 seconds, matching VPP2's
  `t,t+4,...,t+64` video sampling. This is computed directly from body transforms,
  not by summing four unrelated pixel-indexed flow maps.

`valid_mask` marks a known 3D material correspondence. A point may remain valid
when its endpoint is occluded or out of view. `visible_next` is a separate endpoint
visibility test against target depth (5 mm tolerance). The last `dt` rows have
zero flow and false flow validity, rather than repeated fabricated targets.

## Why original states cannot simply be attached to this dataset

The current release is `yuanty/LIBERO-fastwam`, processed with MuJoCo 3.3.2.
Parquet has actions, end-effector, joint and gripper observations, but no object
states, full simulator states, original demo IDs, or no-op index mapping.

The pipeline retrieves compact state/action metadata from the original LIBERO
HDF5, using checked HTTP byte ranges instead of downloading its RGB arrays.
The pinned source is:

```text
yifengzhu-hf/LIBERO-datasets
revision f13aa24a3da8c43c7225569f28c562979fa0e35a
```

Episodes are matched by their entire ordered action sequence. Gripper conversion
is checked explicitly: simulator `-1=open,+1=close`; this LeRobot release has
`1=open,0=close`. Ambiguous matches fail. Episode numbering is never used as the
correspondence after unsuccessful trajectories and no-ops have been filtered.

Original simulator states describe the older recorded trajectory. Therefore the
pipeline restores only the matched original initial state, settles for ten dummy
steps, and replays the retained original controls with robosuite 1.4.0 and MuJoCo
3.3.2. It compares **every** robot joint and gripper observation with parquet,
then verifies **every** rendered frame against the two existing videos.
The default maximum absolute proprio error is 1e-5; RGB mean absolute error must
be at most 3/255 at every frame. A failed check aborts publication of the sidecar.

There is an internal MuJoCo timing detail: `mj_step` integrates `qpos` after
updating image-time derived transforms. Re-running `mj_forward` on the final
`qpos` can shift moving surfaces relative to the recorded image. Replay snapshots
therefore preserve image-time body, geom, camera and light transforms as well as
the full simulator state. Extraction restores those transforms for rendering.

## Environment

The replay runtime is isolated from the training and PRO evaluation packages.
It reuses `../miniconda/envs/vpp2-libero-pro` for MuJoCo 3.3.2, HDF5, PyAV and
other dependencies, while loading robosuite 1.4.0 from a separate directory.
It uses the existing standard LIBERO source/assets under
`third_party/LIBERO-PRO-assets/`, not PRO's perturbed tasks or robosuite 1.5.2.

```bash
bash scripts/libero/setup_motion_runtime.sh
source scripts/libero/activate_motion_env.sh
```

The setup script verifies the PyPI wheel SHA256. `VPP2_MOTION_RUNTIME` can override
the isolated runtime directory. Network downloads need the workspace's normal
network/proxy configuration. Both requests and curl retain TLS verification.

## Generate episodes

Run from the repository root. The pipeline is sequential, accepts an episode
list, and skips existing complete outputs only after checking their contract:

```bash
source scripts/libero/activate_motion_env.sh
python -m vpp2.libero.motion_pipeline \
  --dataset data/libero/libero_object_no_noops_lerobot \
  --episodes 1 2
```

Use `--source-dir /path/to/original/libero_object` to use local original task
HDF5s instead of the pinned Hugging Face files. `--all-episodes` selects the entire
suite. The other three suites use the same command with another dataset root.
Source matching and replay fail closed when a trajectory cannot be reproduced.
The initial implementation has been validated on Object episode 0; the remaining
episodes must pass these checks individually.

The lower-level entrypoints are:

```text
python -m vpp2.libero.motion_source --help
python -m vpp2.libero.motion_replay --help
python -m vpp2.libero.motion_extract --help
```

The verified episode-0 replay is already cached at:
`data/libero_motion_sources/replay_object_episode_000000_v2.hdf5`.

## Files

```text
libero_object_no_noops_lerobot/
├── data/                                  # existing parquet
├── videos/                                # existing RGB
├── motions/
│   ├── chunk-000/episode_000000.hdf5
│   ├── normalized_clips/episode_000000_start040_dt4.npz
│   └── visualizations/episode_000000/
│       ├── motion_preview.png
│       └── motion_viewer.html
└── meta/motions/
    ├── schema.json
    ├── episode_000000.json
    └── episode_000000_validation.json
```

HDF5 stores metric float32 arrays and reversible normalization parameters:

```text
frame_index, timestamp, source_frame_index
reference_camera_to_sim_world
normalization/{center,scale}
body_rotations, body_positions
image/ and wrist_image/
    point_map                [T,H,W,3], metres in shared reference frame
    depth                    [T,H,W], optical-axis metric depth
    valid_mask               [T,H,W]
    geom_id, body_id          [T,H,W]
    rgb                      [T,H,W,3], aligned re-rendered RGB
    camera_to_reference      [T,4,4]
    intrinsic_native, sample_pixel_x, sample_pixel_y
    rgb_mae                  [T]
    flow_dt1/ and flow_dt4/
        scene_flow           [T,H,W,3], metres of displacement
        valid_mask           [T,H,W]
        visible_next         [T,H,W]
```

The first validated sidecar is about 206 MiB for 145 frames. Extrapolating this
one episode by frame count gives roughly 93 GiB for Object and 385 GiB for all
four suites, before visualization/clip exports. Compression and scene complexity
affect the actual total. Full-dataset preprocessing was not launched in this check.

## Normalization and reading

Following MotionCrafter, over all valid points of a selected sequence:

```text
mu = mean(X)
S  = mean(||X - mu||_2) + epsilon
X_normalized = (X - mu) / S
V_normalized = V / S
```

Flow is not centered, independently standardized per axis, or clipped to [-1,1].
One shared scalar preserves direction and the point/flow scale relationship.
Both cameras share the same normalization. Metric values stay on disk so that
normalization can be recomputed on the actual training clip.

For the separate `libero_flow` training pipeline, normalization follows
`flow.motion_cameras` as of 2026-10-26: the default `[image]` computes the
center and scale from only the eight sampled external-camera point maps.
The wrist camera does not participate in these statistics. Both HDF5 and
LZ4 readers use this rule; the metric files do not need regeneration.
The low-level `load_motion_clip` API below retains its dual-view convention.

```python
from vpp2.libero.motion_data import load_motion_clip

clip = load_motion_clip(
    "data/libero/libero_object_no_noops_lerobot/motions/chunk-000/episode_000000.hdf5",
    start=40, length=17, stride=4, normalize="clip",
)
flow = clip["cameras"]["image"]["scene_flow"]       # [17,224,224,3]
mask = clip["cameras"]["image"]["flow_valid_mask"] # final frame is unsupervised
```

`normalize="episode"` uses the saved episode center/scale; `"none"` returns
metric coordinates. The normalized NPZ example is an actual 17-frame export at
frames 40,44,...,104. Its valid point centroid is within 2e-10 of zero and mean
radius is 0.999999978. Normalization derived from future geometry belongs to
supervision targets; do not expose those statistics as inference-time inputs.

This provides preprocessing and a motion reader. Adding a geometry head and
mixing flow-only episodes into training are separate experiments. These labels
are simulator ground truth generated through replay of complete demonstrations;
the eventual label-budget study should state that provenance.

## Visualization and validation

```bash
source scripts/libero/activate_motion_env.sh
python -m vpp2.libero.motion_visualize \
  --motion data/libero/libero_object_no_noops_lerobot/motions/chunk-000/episode_000000.hdf5 \
  --output-dir data/libero/libero_object_no_noops_lerobot/motions/visualizations/episode_000000

python -m pytest -q checks/test_libero_motion.py
python checks/validate_libero_motion.py \
  --motion data/libero/libero_object_no_noops_lerobot/motions/chunk-000/episode_000000.hdf5 \
  --output data/libero/libero_object_no_noops_lerobot/meta/motions/episode_000000_validation.json \
  --normalized-clip data/libero/libero_object_no_noops_lerobot/motions/normalized_clips/episode_000000_start040_dt4.npz
```

The HTML viewer is standalone and works offline: rotate/zoom the scene, select
either camera or both, play/scrub the timeline, and inspect RGB point clouds and
colored motion segments. Segment multiplier 1 represents the true displacement.
The optional multiplier is a visualization control, not a change to stored labels.
It follows the paper's representation and visual idiom without its live Viser
server dependency. Camera transforms are not applied again to already canonical
point maps.

## Initial validated result: Object episode 0 (2026-10-22)

- 145 retained frames, matched to original `demo_0`, with indices 0–143 and 147.
- Entire replay: maximum joint error 1.19e-7 rad, gripper error 1.86e-9 m.
- RGB MAE: external mean/max 2.019/2.417; wrist mean/max 2.174/2.916, on 0–255.
- Image orientation independently verified as a horizontal flip relative to
  native MuJoCo rendering for both views.
- Maximum rigid endpoint consistency residual <6.7e-8 m.
- Maximum static-background 3D flow <1.5e-15 m, including the wrist view whose
  camera moves by up to 0.305 m.
- Both intervals' terminal masks pass. Seven geometry/source unit tests pass.
- The PNG was visually inspected; the standalone HTML's draw/play/scrub/drag
  handlers were executed under Node with a canvas stub. A browser rendering
  check has not been run.

## Full generation launched 2026-10-22

The local supervisor runs all 1,712 episodes (277,713 retained frames) across
Object (457), Goal (433), Spatial (434), and LIBERO-10 (388). It processes one
episode subprocess at a time, retries failures once, and records unresolved
failures without publishing mismatched labels. Completed sidecars are verified
and skipped on restart. A filesystem lock prevents duplicate supervisors.
LIBERO-10 source filenames are resolved from BDDL task definitions, preserving
scene prefixes absent from the natural-language task names.

Run directory: `runs/libero_motion_full_20261022/`; inspect `status.json`,
`supervisor.log`, `events.jsonl`, and per-episode logs. A status of
`finished_with_failures` is not full completion: inspect and resolve every
entry in `failed`, then rerun the same supervisor command.

```bash
source scripts/libero/activate_motion_env.sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u scripts/libero/generate_all_motions.py \
  --run-dir runs/libero_motion_full_20261022
```

Definition audit: MotionCrafter v2 §3.1 equations (1–2) and Appendix A equation
(16). Source-pixel material-point forward displacements and canonical
normalization agree. The explicit multi-camera extension uses the initial
agentview coordinate system and one shared normalization over both views;
this differs from the paper's monocular input convention. `flow_dt4` is the
forward flow between consecutive frames of the stride-4 sampled video, not
four same-pixel dt1 vectors added together. On-disk arrays remain metric;
`load_motion_clip(normalize="clip")` normalizes the selected training clip
and excludes its final flow target.

## Replay correction and expanded validation (2026-10-23)

The first full run exposed two reconstruction errors; merely passing the
robot joint check was insufficient to validate dense labels:

- Fixture root poses are randomized in `model.body_pos/body_quat` on reset,
  outside the flattened simulator state. Reproduce `seed(0)` once per task
  and reset `original_demo_index + 1` times, including discarded original
  demos. LeRobot episode indices cannot substitute for original demo indices.
- Robosuite camera observables sample 2 ms before the control step ends.
  Save body/camera transforms inside the camera sensor callback, together
  with the then-current model site/geom/material RGBA (e.g. stove burner
  visibility). Saving end-of-step transforms or the final model appearance
  gives wrong wrist images even with exact joint observations.

Replay version 3 implements both corrections. The camera callback records
geometry and returns placeholder RGB; the real RGB/depth/segmentation are
rendered later from those captured transforms. This avoids rendering twice,
while retaining the original observable schedule. The two camera sample
states and times must match exactly. `image_sim_time` records actual capture
times; dataset-relative timestamps remain at 20 Hz.

No-op indices must equal the original regeneration rule: arm action norm
below `1e-4`, AND the gripper is unchanged from the last retained action
(first action has no previous gripper). Removed commands are never stepped;
gripper changes remain. The source controls retain their original float64
precision. Neither the proprio `1e-5` threshold nor RGB MAE `3.0/255` is raised.

Old motion sidecars are archived under `motions/legacy_replay_v*/` on revisit;
versioned replay caches prevent reuse of old snapshots. The corrected full
run uses `runs/libero_motion_suites_20261023_v3/<suite>/`. A temporary
`data/libero_motion_sources/validation_in_progress` marker holds production
workers while checks are in progress.

Expanded audit inputs and per-episode results live in
`runs/motion_validation_20261023/`: `noop_audit.json`, `selection.json`, and
`v3/<suite>/episode_*.json`. The selection covers 38 full trajectories from
17 currently cached tasks across all four suites; it deliberately includes
heavy no-op filtering, late original demos, and previous failures. This is a
stratified diagnostic sample, not proof that all 40 tasks / 1,712 episodes pass.

```bash
source scripts/libero/activate_motion_env.sh
python checks/validate_motion_replay_samples.py --suite spatial
python checks/summarize_motion_validation.py v3
```

Completed expanded validation (2026-10-23): all 627 cached source index
sequences matched the canonical no-op rule. Of 38 selected trajectories
(6,724 retained frames, 119 removed commands), 37 passed strict replay and
full-trajectory two-camera RGB checks: object 9/9, goal 8/9, spatial 11/11,
LIBERO-10 9/9. Among passing trajectories, maximum per-frame RGB MAE was
2.791255 (limit 3.0), and maximum joint error was 1.955313e-7 rad (limit
1e-5). These are diagnostic sample results, not a population success estimate.

Goal episode 108 (original demo 49) failed at frame 108: joint error
3.108861e-5 rad and gripper error 1.431466e-5 m. It has no removed no-ops.
Both float64 and float32 action replay diverged late in this contact
trajectory; the precise numerical cause remains unresolved. It must not
produce accepted motion labels. Production retains strict fail-closed gates
and continues with subsequent episodes when an episode fails.

An end-to-end dense-flow check on spatial episode 9 also passed: maximum
rigid endpoint error 6.60e-8 m, static flow below 2.62e-16 m, and normalized
clip mean radius 0.99999998. See
`runs/motion_validation_20261023/dense_validation.json` and `v3/summary.json`.
