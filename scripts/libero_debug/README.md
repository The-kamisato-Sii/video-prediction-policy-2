# Archived LIBERO debug launchers

Historical benchmark recipes, retained for reproducibility only:

- `train_8h200.sh`: configurable original ZeRO2, batch 16, accumulation 2.
- `train_8h200_zero1.sh`: eight-GPU ZeRO1 comparison wrapper.
- `train_8h200_zero1_first_frame.sh`: explicit first-frame comparison wrapper.

These resolve the repository root from their own paths. Source `../libero/activate_env.sh` first. Their shared model/data defaults follow the current code; reproduce historical measurements using the corresponding Git revision.

New runs should use `../libero/train_action.sh` (8 GPUs) or the three production 16-GPU entries in `../libero/`. Moving these files does not change the queued production commands.
