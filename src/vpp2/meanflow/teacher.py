"""Frozen teacher trajectories, preserving scheduler dtype and negative Euler deltas."""

import torch


@torch.no_grad()
def rollout(model, noise, condition, steps=10, shift=1.0):
    times, deltas = model.infer_action_scheduler.build_inference_schedule(
        steps, noise.device, noise.dtype, shift_override=shift
    )
    # Keep the actual floating-point sigmas produced by the scheduler, not step indices.
    grid, _ = model.infer_action_scheduler.build_inference_schedule(
        steps, noise.device, torch.float32, shift_override=shift
    )
    sigmas = torch.cat((grid / model.infer_action_scheduler.num_train_timesteps, grid.new_zeros(1)))
    states = [noise]
    for time, delta in zip(times, deltas):
        velocity = model._predict_action_noise_with_cache(
            latents_action=states[-1], timestep_action=time.expand(noise.shape[0]), **condition
        )
        states.append(model.infer_action_scheduler.step(velocity, delta, states[-1]))
    return torch.stack(states), sigmas
