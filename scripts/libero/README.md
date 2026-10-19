# LIBERO launchers

Production entry points (two nodes, eight H200 each, microbatch16, accumulation1):

- `train_16h200.sh`: FM baseline.
- `train_meanflow_16h200.sh`: fresh-init MeanFlow.
- `train_direct_action_16h200.sh`: fresh-init clean-action regression.

Source `activate_env.sh` before invoking them. Each accepts `--dry-run` and OmegaConf `key=value` overrides. Qizhi supplies PET_NNODES / PET_NPROC_PER_NODE / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT. Commands and the comparison contract are in `../../docs/libero_one_step.md`.

`train_one_step.sh` implements the shared one-step launcher; the method-specific scripts only select the method and enforce the production layout. `train_8h200_zero1.sh` is the baseline one-node b32 recipe. `train_8h200_zero1_first_frame.sh` remains a compatibility alias with explicit first-frame flags; first-frame decoding is now already the Action default. `train_8h200.sh` is the configurable low-level local-path wrapper. `train_action.sh` and `train_video.sh` are portable low-level entries.

Only tracked launchers belong here. Keep run-specific probes/logs under ignored `runs/`. Replace scripts atomically when editing; do not rewrite a file that an executing shell may still be reading in place. Production launchers exec the training process, preserve its exit code, and hold a per-run/per-node lock.
