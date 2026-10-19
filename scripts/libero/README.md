# LIBERO production launchers

Source `activate_env.sh`, then run one entry on every node:

- `train_action.sh`: Flow Matching baseline.
- `train_meanflow.sh`: MeanFlow.
- `train_direct_action.sh`: direct action MSE.

All entries use ZeRO1, first-frame/uint8 decoding, four workers per GPU, global batch 256 and gradient accumulation 1. `topology.sh` reads Qizhi PET_* allocation variables (or explicit NNODES/NPROC_PER_NODE); outside the platform it counts visible GPUs with PyTorch and defaults to one node. Local visible GPU count is checked again before training. Each GPU gets 256 / total GPU count samples: 8 GPUs → 32, 16 → 16, 32 → 8. Non-divisors of 256 and batch CLI overrides fail before training. Batch is not reduced automatically on OOM.

For manual multi-node runs set NNODES, NODE_RANK and MASTER_ADDR on each node. Qizhi supplies these through PET_*. `--dry-run` validates the resolved recipe without starting training.

`train_baseline.sh`, `train_one_step.sh` and `topology.sh` are shared dependencies. Preparation, Video post-training and evaluation tools remain here; the automatic batch contract applies to the three Action training methods. Historical fixed-topology/debug launchers are in `../libero_debug/`. Keep run-specific logs under ignored `runs/`.
