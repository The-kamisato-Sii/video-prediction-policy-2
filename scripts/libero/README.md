# LIBERO production launchers

Source `activate_env.sh` before training. All Action recipes default to first-frame decoding, uint8 decoding without redundant conversions, four workers per GPU, ZeRO1, and global batch 256.

- `train_16h200.sh`: FM baseline, two nodes × eight GPUs, batch 16 per GPU, accumulation 1.
- `train_meanflow_16h200.sh`: MeanFlow, same 16-GPU layout.
- `train_direct_action_16h200.sh`: direct action MSE, same 16-GPU layout.
- `train_action.sh`: confirmed single-node FM recipe, eight GPUs, batch 32 per GPU, accumulation 1.

`train_baseline.sh` is the shared baseline path/config launcher; `train_one_step.sh` is the shared one-step launcher. Keep these dependencies here. Data preparation, Video post-training, evaluation and environment utilities are also retained.

Historical ZeRO2 and first-frame comparison scripts are archived in `../libero_debug/`. Do not use them for new production runs. Commands for the 16-GPU experiments are in `../../docs/libero_one_step.md`.

Production launchers accept `--dry-run` and OmegaConf overrides. Keep run-specific probes and logs under ignored `runs/`. Replace running scripts atomically; production entry points preserve exit codes and hold per-node locks.
