"""Run the official real-data LIBERO trainer with bounded correctness checks.

Use VPP2_TRAIN_MODULE=checks.libero_training_audit with train_8h200.sh.
Full tensor checksums happen outside the measured training loop.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from vpp2.libero import trainer as trainer_module
from vpp2.libero.cli import main


def digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


class AuditedLiberoTrainer(trainer_module.LiberoTrainer):
    def train(self):
        model = self.accelerator.unwrap_model(self.model)
        assert self.train_mode == "action_only"
        assert self._deepspeed_zero_stage() == 2
        allowed = {id(p) for module in (model.action_expert, model.proprio_encoder)
                   for p in module.parameters()}
        assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
        assert not any(p.requires_grad for p in model.video_expert.parameters())
        assert model.mot.training and model.action_expert.use_gradient_checkpointing
        assert not model.video_expert.training
        # Every rank checks the entire frozen Video before and after training.
        before_video = digest(model.video_expert)
        before_action = digest(model.action_expert)
        before_proprio = digest(model.proprio_encoder)
        started_step = self.global_step
        engine_started_step = self.model.global_steps
        gradient_norms = {}
        handles = []
        for name, parameter in (("action", model.action_expert.head.weight),
                                ("proprio", model.proprio_encoder.weight)):
            def hook(gradient, name=name):
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Non-finite {name} gradient")
                gradient_norms[name] = float(gradient.float().norm())
            handles.append(parameter.register_hook(hook))
        self.accelerator.wait_for_everyone()
        super().train()
        for handle in handles:
            handle.remove()
        assert self.global_step > started_step
        assert self.model.global_steps - engine_started_step == self.global_step - started_step
        assert all(p.grad is None for p in model.video_expert.parameters())
        assert digest(model.video_expert) == before_video, "Frozen Video changed"
        after_action = digest(model.action_expert)
        after_proprio = digest(model.proprio_encoder)
        assert after_action != before_action, "Action did not update"
        assert after_proprio != before_proprio, "Proprio projection did not update"
        assert all(gradient_norms.get(name, 0) > 0 for name in ("action", "proprio"))
        local = dict(
            rank=self.accelerator.process_index, starting_step=started_step,
            final_step=self.global_step, frozen_video_unchanged=True,
            engine_starting_step=engine_started_step, engine_final_step=self.model.global_steps,
            action_activation_checkpointing=True, video_eval_mode=True,
            frozen_video_sha256=before_video, frozen_video_grad_none=True,
            action_updated=True, proprio_updated=True, finite_gradient_norms=gradient_norms,
            action_sha256=after_action, proprio_sha256=after_proprio,
            zero_stage=2, micro_batch=self.batch_size,
            accumulation=self.gradient_accumulation_steps,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_video_parameters=sum(p.numel() for p in model.video_expert.parameters()),
        )
        ranks = [None] * self.accelerator.num_processes
        dist.all_gather_object(ranks, local)
        for key in ("action_sha256", "proprio_sha256", "frozen_video_sha256"):
            assert len({r[key] for r in ranks}) == 1, f"Rank replicas differ: {key}"
        if self.accelerator.is_main_process:
            (Path(self.output_dir) / "training_audit.json").write_text(
                json.dumps(dict(status="PASS", replicas_identical=True, ranks=ranks), indent=2) + "\n"
            )
        self.accelerator.wait_for_everyone()


if __name__ == "__main__":
    trainer_module.LiberoTrainer = AuditedLiberoTrainer
    main()
