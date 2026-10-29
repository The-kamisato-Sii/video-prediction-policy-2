"""Real-data ZeRO-1 startup gate. Keeps the 30k schedule, bounds only the loop."""
import json
import math
import time
from pathlib import Path
import torch
from libero_ablation_smoke import AblationSmokeTrainer

class IMFSmokeTrainer(AblationSmokeTrainer):
    def train(self):
        model=self.accelerator.unwrap_model(self.model)
        assert model.meanflow_config['objective']=='data_improved_meanflow'
        assert len(model.action_expert.blocks)==40
        auxiliary=model.meanflow_config['imf']['variant']=='auxiliary'
        norms=[];consume=model.consume_imf_gradient_metrics
        def capture():
            metrics=consume()
            assert all(math.isfinite(v) for v in metrics.values()),'Nonfinite branch gradient'
            norms.append(metrics)
            return metrics
        model.consume_imf_gradient_metrics=capture
        # Count actual deployment calls separately from training calls.
        aux_calls=[]
        handle=model.action_expert.aux_v_head.register_forward_hook(
            lambda *args:aux_calls.append(model.action_expert.training)) if auxiliary else None
        started=time.monotonic()
        super().train()  # real updates, compact reload, one-NFE deployment
        elapsed=time.monotonic()-started
        if handle:handle.remove()
        assert len(norms)==self.global_step
        for group in ['shared','u_tail']+(['v_tail'] if auxiliary else []):
            assert max(n['imf/local_pre_reduce_grad_norm_'+group] for n in norms)>0,group
        assert not any(training is False for training in aux_calls),'Deployment called auxiliary head'
        # Exercise the real evaluator's rank-zero-only training_loss collectives.
        metrics=self.evaluate()
        if metrics:
            assert all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
        report=dict(rank=self.accelerator.process_index,world_size=self.accelerator.num_processes,
                    variant=model.meanflow_config['imf']['variant'],steps=self.global_step,
                    batch_per_gpu=int(self.cfg.batch_size),schedule_steps=int(self.cfg.max_steps),
                    elapsed_including_reload_inference_seconds=elapsed,gradient_norms=norms,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    compact_reload=True,inference_nfe=1,auxiliary_inference_calls=0,
                    rank0_evaluation=True,evaluation=metrics)
        dest=Path(self.cfg.output_dir)/f'imf_smoke_rank{self.accelerator.process_index}.json'
        dest.write_text(json.dumps(report,indent=2))
        self.accelerator.wait_for_everyone()
        print('IMF_FULLSIZE_SMOKE_PASS',json.dumps({k:v for k,v in report.items() if k not in ['gradient_norms','evaluation']}),flush=True)

if __name__=='__main__':
    import vpp2.libero.trainer
    from vpp2.meanflow.cli import main
    vpp2.libero.trainer.LiberoTrainer=IMFSmokeTrainer
    main()
