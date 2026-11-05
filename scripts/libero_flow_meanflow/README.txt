Shared Flow + Action MeanFlow experiments (2026-11-04)

Matrix: {both, recon} x {adaptive, diagonal, imf_boundary, imf_auxiliary}.
All eight: 32 tokens; action1/8 seed42 manifest (214 episodes); all valid simulator
GT flow (1589 episodes); same fixed motion manifest as previous GT experiments.
4RC is NOT the label source for this matrix. Failed/missing replay flow is skipped
only for the flow records. Action records still follow the original subset list.

Fixed settings: 2 nodes x 8 H200, priority10, 30000 formal optimizer updates,
global batch256, microbatch16, GAS1, LR1e-4, warmup500, same optimizer/LR schedule,
video10k frozen, fresh action_init, seed42, same 12-worker LZ4 loader, original
homogeneous_interleaved global-batch sampler, save5000/eval200. RGB conditions
remain dual camera. Labels use only external camera with its clip mu/radius,
frames0,4,...28 and dt4. MotionCodec retains official scene-flow x10 and latent
scaling. No older source or training script was modified.

Both: action and motion latent use the same MF/iMF variant. Frozen pretrained
MotionVAE maps normalized GT [B,1,6,8,224,224] -> [B,8,8,28,28]. Patch (4,7,7),
stride (4,7,7), no padding -> [B,32,512]. Shared action backbone + indicator and
end-time conditioning. Action head -> [B,32,7]; motion deconvolution -> latent.
Inference: one step from t1,r0, clean=noise-u; frozen VAE decodes motion.
The existing Gaussian latent-velocity skip remains, including in the JVP.

Recon: action uses MF/iMF; motion retains the existing learned-query clean
reconstruction. Random MotionVAE decoder (no pretrained motion weights), frozen
unused encoder/input patch; 0.5*(point_map MSE +100*scene_flow MSE), each masked
and normalized separately. Exact same reconstruction implementation is reused.
One-step action and one-pass reconstructed motion.

Variants exactly inherit configs/libero/train_meanflow*.yaml and train_imf*.yaml:
adaptive: adaptive power0.5,c0.001; diagonal probability0.5, sorted_max time;
diagonal: same with independent diagonal time;
imf_boundary: independent diagonal, predicted u(z,t,t) direction (detached);
imf_auxiliary: independent diagonal, separate copied last8 layers and actual
adapter output heads. Shared prefix32 layers; auxiliary loss weight1. The old
unused ActionDiT linear head/aux head stay frozen. Auxiliary adds parameters,
matching the prior action-only iMF auxiliary design. Recon auxiliary only needs
an auxiliary action head; the reconstruction branch has no auxiliary v loss.

Original vpp2.meanflow.objectives is imported unmodified. A view [B,elements,1]
allows its existing mask and per-sample adaptive reduction to cover either label
shape. Detached JVP and boundary predictions are processed in chunks of4 to
bound temporary KV-cache copies. This is not gradient accumulation: the live
batch/loss/optimizer batch remain16/256. No cross-example normalization is used.
Analytic loss/gradient checks and actual-wrapper comparisons against unchunked
MF/iMF cover >4 rows and completely invalid examples.

Validation: check_math.py, check_adapter.py, prepare_jobs.py, validate_model.py.
Full-backbone B1 tests use real dataset RGB/proprio/labels, exact JVP, backward,
finite/nonzero shared and task gradients, frozen video/codec boundaries,
one-step outputs, and strict checkpoint roundtrip. Each queued job additionally
runs an isolated 8-update 16-GPU smoke test at batch16; the unchanged sampler
first reaches action at update6. Formal30000 starts in a fresh process with
seed42 and the original initialization, never from smoke weights/optimizer.

Dedicated checkpoint format: vpp2.libero_flow.meanflow.v1. Saved meanflow config,
heads, auxiliary tail, and reconstruction codec/queries are strictly restored.
Dedicated inference/evaluate modules use one-step sampling and check first-real-
observation parity with training sample_task. Do not use the old NFE10 evaluator.

Submission payloads, inline commands, config diffs, IDs/status:
runs/libero_flow_meanflow_submit_20261104/
Validation evidence: runs/libero_flow_meanflow_validation_20261104/
New formal runs: runs/libero_flow_meanflow/
