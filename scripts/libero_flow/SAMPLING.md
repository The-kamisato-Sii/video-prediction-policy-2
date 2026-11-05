# Flow/action task-homogeneous batches

Default: `flow.batch_sampling=homogeneous_interleaved`.

Action samples come from the 1/8, 1/4, 1/2 or all-data (1/1) episode manifest.
The all-data manifest includes every baseline episode, including episodes without valid flow.
All eligible flow samples remain included, independently of action membership;
missing flow skips only its flow record. Dataset definitions and losses are unchanged.

Each global optimizer batch uses one task on every rank, including all gradient
accumulation microbatches. Each task pool is independently shuffled every epoch.
Task batches are evenly interleaved in proportion to their batch counts, with a
seeded rotation each epoch; this is not a fixed 1:1 task ratio.

All samples are visited per complete epoch. Each task's final global batch repeats
samples from its shuffled pool to reach global batch256 (at most255 repeats per
task/epoch). No tail samples are dropped. `sampler_audit.json` reports exact counts.
Total optimizer steps30000 and global batch256 remain unchanged. Runs end at the
step budget, so the final epoch can be partial, as before.

`flow.batch_sampling=mixed` reproduces the old within-batch task mixture for
controlled diagnostics. Existing mixed-order checkpoints must retain `mixed` for
exact data-order resume; switching sampling policy changes the training trajectory.
New homogeneous runs support deterministic epoch and microbatch-offset resume.
Do not compare convergence without treating this task scheduling change as part
of the experiment. The original baseline/MeanFlow samplers are untouched.

CPU contract validation:
```
source scripts/libero_flow/activate_env.sh
"$PYTHON_BIN" checks/test_flow_task_sampler.py
```
This uses Accelerate's real BatchSamplerShard for1/2/16ranks and accumulation,
checks coverage, homogeneous global batches, tail accounting and exact resume.
