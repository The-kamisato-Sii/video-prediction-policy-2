"""Checkpointing must be active without unfreezing the Video expert."""

import copy
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from test_meanflow import tiny_model, inputs
from vpp2.libero.action_model import LiberoActionModel
from vpp2.libero.trainer import LiberoTrainer


def test_action_checkpointing_preserves_loss_and_gradients():
    torch.manual_seed(72)
    model = tiny_model(cls=LiberoActionModel, checkpoint=True)
    trainer = LiberoTrainer.__new__(LiberoTrainer)
    trainer.train_mode = "action_only"
    trainer._apply_train_mode(model)
    assert model.mot.training
    assert model.action_expert.training and model.proprio_encoder.training
    assert not model.video_expert.training and not model.vae.training
    allowed = {id(p) for module in (model.action_expert, model.proprio_encoder)
               for p in module.parameters()}
    assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
    reference = copy.deepcopy(model)
    reference.mot.mot_checkpoint_mixed_attn = False
    reference.action_expert.use_gradient_checkpointing = False

    def evaluate(m):
        torch.manual_seed(99)
        sample = inputs(m)
        loss, _ = m._training_loss_action_only(
            latents=torch.randn(2, 16, 2, 2, 2),
            noisy_action=sample.pop("action"), target_action=torch.randn(2, 4, 3),
            timestep_video=torch.full((2,), 1000.0),
            timestep_action=torch.tensor([250.0, 750.0]), action_is_pad=None,
            **sample,
        )
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}

    with patch("torch.utils.checkpoint.checkpoint", wraps=checkpoint) as spy:
        actual_loss, actual_grads = evaluate(model)
    # Two Action layers, each with mixed-attention and post-attention checkpoint.
    # Frozen Video prefill stays under no_grad and does not use checkpointing.
    assert spy.call_count == 4
    expected_loss, expected_grads = evaluate(reference)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-5, atol=1e-7)
    assert actual_grads["proprio_encoder.weight"].norm() > 0
    assert all(p.grad is None for p in model.video_expert.parameters())
