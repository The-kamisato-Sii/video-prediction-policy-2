"""Reuse baseline optimizer/LR/resume machinery; isolate loader and trainability."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from vpp2.libero.trainer import LiberoTrainer
from vpp2.utils.samplers import ResumableEpochSampler
from .data import collate_flow_action
from .sampler import HomogeneousTaskSampler


class LiberoFlowTrainer(LiberoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Carry accumulation across epoch boundaries instead of emitting a
        # smaller optimizer batch when a loader length is not divisible by GAS.
        # This changes only this process's Accelerator state, never baseline code.
        self.accelerator.gradient_state.plugin_kwargs["sync_with_dataloader"] = False

    @staticmethod
    def _validate_preflight_config(cfg):
        return None  # New shape/missing-label contracts are checked by validate.py.

    def _apply_train_mode(self, model):
        model.configure_trainability()

    def _build_loader(self, dataset, worker_init_fn=None):
        prefetch = int(self.cfg.flow.get("prefetch_factor", 1))
        if prefetch < 1:
            raise ValueError("flow.prefetch_factor must be positive")
        policy = str(self.cfg.flow.get("batch_sampling", "homogeneous_interleaved"))
        options = dict(dataset=dataset, seed=self.seed, batch_size=self.batch_size,
                       num_processes=self.accelerator.num_processes)
        if policy == "homogeneous_interleaved":
            self.train_sampler = HomogeneousTaskSampler(**options,
                accumulation_steps=int(self.cfg.gradient_accumulation_steps))
            if self.accelerator.is_main_process:
                path = Path(self.output_dir) / "sampler_audit.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.train_sampler.audit, indent=2) + "\n")
        elif policy == "mixed":
            self.train_sampler = ResumableEpochSampler(**options)
        else:
            raise ValueError(f"Unknown flow.batch_sampling: {policy}")
        return DataLoader(dataset, batch_size=self.batch_size, sampler=self.train_sampler,
            num_workers=self.num_workers, collate_fn=collate_flow_action,
            pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
            **({"prefetch_factor": prefetch} if self.num_workers else {}))

    def _save_weights_checkpoint(self, step_tag, *, state_dict_overrides=None):
        path = str(Path(self.weights_dir) / f"{step_tag}.pt")
        self.accelerator.unwrap_model(self.model).save_flow_checkpoint(path,
            step=self.global_step, resolved_cfg=self.cfg)
        return path
