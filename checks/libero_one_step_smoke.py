"""Full-size real-data smoke; retain the production LR schedule, shorten only the loop."""
import os
from vpp2.libero.trainer import LiberoTrainer

class SmokeTrainer(LiberoTrainer):
    def train(self):
        self.max_steps = int(os.environ.get("VPP2_SMOKE_STEPS", "4"))
        result = super().train()
        assert self.global_step == self.max_steps
        print(f"ONE_STEP_SMOKE_PASS rank={self.accelerator.process_index} step={self.global_step}", flush=True)
        return result

if __name__ == "__main__":
    import vpp2.libero.trainer
    from vpp2.meanflow.cli import main
    vpp2.libero.trainer.LiberoTrainer = SmokeTrainer
    main()
