"""Bound a real trainer run without shortening its configured LR schedule."""
import json
import os
import faulthandler
import inspect
import signal
from pathlib import Path

from vpp2.libero.trainer import LiberoTrainer
from vpp2.meanflow.objectives import data_loss
from vpp2.meanflow.model import MeanFlowMixin

faulthandler.register(signal.SIGUSR1, all_threads=True)
assert 'distributed' in inspect.signature(data_loss).parameters
assert any(isinstance(x, tuple) and 'distributed' in x
           for x in MeanFlowMixin.training_loss.__code__.co_consts)
print('ADAPTIVE_AUDIT: loaded loss and model both support distributed=False', flush=True)

_original_train = LiberoTrainer.train
_original_evaluate = LiberoTrainer.evaluate


def _audited_evaluate(self):
    print(f'EVAL_START rank={self.accelerator.process_index}', flush=True)
    result = _original_evaluate(self)
    print(f'EVAL_END rank={self.accelerator.process_index}', flush=True)
    return result


def _bounded_train(self):
    configured_max = self.max_steps
    resumed_step = self.global_step
    limit = int(os.environ['VPP2_SMOKE_STOP_STEP'])
    if not self.global_step < limit < configured_max:
        raise ValueError('Smoke stop must follow resumed step and precede configured budget')
    model = self.accelerator.unwrap_model(self.model)
    original_loss = model.training_loss
    def audited_loss(*args, **kwargs):
        if not model.action_expert.training:
            print(f'EVAL_LOSS_START rank={self.accelerator.process_index} action_training=False', flush=True)
        result = original_loss(*args, **kwargs)
        if not model.action_expert.training:
            print(f'EVAL_LOSS_END rank={self.accelerator.process_index}', flush=True)
        return result
    model.training_loss = audited_loss
    assert model.meanflow_config['loss_weighting'] == 'adaptive'
    assert not any(p.requires_grad for p in model.video_expert.parameters())
    assert all(p.requires_grad for p in model.action_expert.parameters())
    assert all(p.requires_grad for p in model.proprio_encoder.parameters())
    # The first run verifies the full save; the resumed run verifies two updates.
    # Avoid writing a second identical-size diagnostic state to shared storage.
    if resumed_step:
        self.save_final_checkpoint = False
    self.max_steps = limit  # Scheduler and MeanFlow budget were already built for 30k.
    _original_train(self)
    assert self.global_step == limit
    if self.accelerator.is_main_process:
        Path(self.output_dir, f'adaptive-smoke-{limit}.json').write_text(json.dumps(dict(
            status='PASS', step=self.global_step, schedule_steps=configured_max,
            resumed_step=resumed_step,
            meanflow=model.meanflow_config, checkpoint=self.last_checkpoint_info,
            world_size=self.accelerator.num_processes, batch_per_gpu=self.batch_size,
        ), indent=2))
    self.accelerator.wait_for_everyone()


LiberoTrainer.train = _bounded_train
LiberoTrainer.evaluate = _audited_evaluate
if __name__ == '__main__':
    from vpp2.meanflow.cli import main
    main()
