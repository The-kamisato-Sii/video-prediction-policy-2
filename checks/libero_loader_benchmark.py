"""Bounded real 8-GPU training; preserve the official 30k-step LR schedule."""
import os
import torch
from vpp2.libero import trainer as trainer_module
from vpp2.libero.cli import main


class BenchmarkTrainer(trainer_module.LiberoTrainer):
    def train(self):
        # The optimizer/scheduler have already been built for cfg.max_steps=30000.
        assert self.accelerator.num_processes == 8
        assert self._deepspeed_zero_stage() == 1
        assert self.batch_size == 32 and self.gradient_accumulation_steps == 1
        self.max_steps = int(os.environ.get('VPP2_BENCH_STEPS', '56'))
        self.save_every = 0
        self.save_final_checkpoint = False
        self.eval_every = 0
        super().train()


if __name__ == '__main__':
    torch.manual_seed(42)
    trainer_module.LiberoTrainer = BenchmarkTrainer
    main()
