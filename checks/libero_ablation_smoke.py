"""Real-data two-rank diagnostic; production schedule, independently shortened loop."""
import json
import os
from pathlib import Path
import torch
from vpp2.libero.trainer import LiberoTrainer


class AblationSmokeTrainer(LiberoTrainer):
    def train(self):
        assert int(self.cfg.max_steps) == 30000  # Optimizer/schedule already constructed.
        self.max_steps = int(os.environ.get("VPP2_SMOKE_STEPS", "4"))
        # Exercise the deployable checkpoint here; optimizer/scheduler round trips
        # are covered by test_full_optimizer_query_round_trip. Avoid repeatedly
        # serializing the identical frozen 16B Video expert in a four-step probe.
        self.save_final_checkpoint = False
        model = self.accelerator.unwrap_model(self.model)
        assert not any(p.requires_grad for p in model.video_expert.parameters())
        query = getattr(model.action_expert, "direct_action_queries", None)
        query_grads = []
        if query is not None:
            assert query.requires_grad and query.count_nonzero() == 0
            def inspect_gradient(grad):
                assert torch.isfinite(grad).all(), "Nonfinite query gradient"
                query_grads.append(float(grad.detach().float().norm()))
            handle = query.register_hook(inspect_gradient)
        torch.cuda.reset_peak_memory_stats()
        super().train()
        assert self.global_step == self.max_steps
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_weights_checkpoint(f"step_{self.global_step:06d}")
        self.accelerator.wait_for_everyone()
        if query is not None:
            handle.remove()
            assert query_grads and max(query_grads) > 0
            assert torch.isfinite(query).all() and query.count_nonzero() > 0
            saved_query = query.detach().clone()
            gathered = self.accelerator.gather(query.detach())
            for other in gathered.chunk(self.accelerator.num_processes):
                torch.testing.assert_close(saved_query, other, rtol=0, atol=0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        path = Path(self.weights_dir) / f"step_{self.global_step:06d}.pt"
        assert model.load_checkpoint(path, load_video_from_action_checkpoint=False) == self.global_step
        if query is not None:
            torch.testing.assert_close(query, saved_query, rtol=0, atol=0)
        model.eval()
        seen = []
        def inspect_input(module, args):
            if model.action_sampler == "direct_action":
                expected = model.action_expert.direct_action_input(args[0])
                torch.testing.assert_close(args[0], expected, rtol=0, atol=0)
            seen.append(1)
        handle = model.action_expert.action_encoder.register_forward_pre_hook(inspect_input)
        with torch.no_grad():
            result = model.infer_action(
                None, torch.zeros(1, 3, 224, 448, device=model.device), 32,
                context=torch.zeros(1, 1, model.text_dim, device=model.device, dtype=model.torch_dtype),
                context_mask=torch.ones(1, 1, device=model.device, dtype=torch.bool),
                proprio=torch.zeros(1, 8, device=model.device, dtype=model.torch_dtype),
                num_inference_steps=1, seed=42,
            )
        handle.remove()
        assert len(seen) == 1 and torch.isfinite(result["action"]).all()
        report = dict(rank=self.accelerator.process_index, steps=self.global_step,
                      schedule_steps=int(self.cfg.max_steps), batch_per_gpu=int(self.cfg.batch_size),
                      peak_allocated_gib=peak, compact_reload=True, inference_nfe=len(seen),
                      query_gradient_norms=query_grads, config=model.meanflow_config)
        (Path(self.cfg.output_dir) / f"smoke_rank{self.accelerator.process_index}.json").write_text(json.dumps(report, indent=2))
        self.accelerator.wait_for_everyone()
        print(f"ABLATION_SMOKE_PASS rank={self.accelerator.process_index} step={self.global_step}", flush=True)


if __name__ == "__main__":
    import vpp2.libero.trainer
    from vpp2.meanflow.cli import main
    vpp2.libero.trainer.LiberoTrainer = AblationSmokeTrainer
    main()
