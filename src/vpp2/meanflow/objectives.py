"""MeanFlow math in normalized clean=0/noise=1 time; no encoders or postprocessing."""

import torch
from torch import nn

from .loss_config import loss_settings
from .experiment_config import interval_settings


@torch.no_grad()
def loss_diagnostics(per_sample, weighted, weights, has_valid, gap, correction, is_pad,
                     *, t, distributed=True):
    """Global valid-sample statistics; every rank enters the same two collectives."""
    diagonal = has_valid & (gap == 0)
    offdiagonal = has_valid & (gap > 0)
    valid = valid_elements(correction, is_pad)
    off_valid = valid & expand_time(offdiagonal, correction)
    # Fixed-size sums/counts: even an empty group participates in both collectives.
    distribution = {
        "mf/t_diagonal_sum": torch.where(diagonal, t, 0).sum(),
        "mf/t_offdiagonal_sum": torch.where(offdiagonal, t, 0).sum(),
        "mf/gap_offdiagonal_sum": torch.where(offdiagonal, gap, 0).sum(),
        "mf/correction_offdiagonal_squared_sum": torch.where(off_valid, correction.float().square(), 0).sum(),
        "mf/offdiagonal_valid_elements": off_valid.sum(),
    }
    for group, rows in (("diagonal", diagonal), ("offdiagonal", offdiagonal)):
        for quantity, values in (("t", t), ("gap", gap)):
            for index in range(5):
                bucket = (values >= index / 5) & ((values < (index + 1) / 5) if index < 4 else (values <= 1))
                distribution[f"mf/{quantity}_{group}_bin{index}_count"] = (rows & bucket).sum()
    sums = torch.stack([
        torch.where(has_valid, per_sample, 0).sum(),
        torch.where(has_valid, weighted, 0).sum(),
        torch.where(diagonal, per_sample, 0).sum(),
        torch.where(offdiagonal, per_sample, 0).sum(),
        torch.where(has_valid, weights, 0).sum(),
        has_valid.sum(), diagonal.sum(), offdiagonal.sum(),
        torch.where(valid, correction.float().square(), 0).sum(), valid.sum(),
        *distribution.values(),
    ]).float()
    maximum = torch.where(has_valid, weights, 0).max()
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    count, diag_count, off_count = sums[5:8]
    metrics = {
        "mf/raw_mse": sums[0] / count.clamp_min(1),
        "mf/weighted_mse": sums[1] / count.clamp_min(1),
        "mf/raw_mse_diagonal": sums[2] / diag_count.clamp_min(1),
        "mf/raw_mse_offdiagonal": sums[3] / off_count.clamp_min(1),
        "mf/weight_mean": sums[4] / count.clamp_min(1),
        "mf/weight_max": maximum,
        "mf/valid_samples": count,
        "mf/diagonal_samples": diag_count,
        "mf/offdiagonal_samples": off_count,
        "mf/correction_rms": (sums[8] / sums[9].clamp_min(1)).sqrt(),
        "mf/t_diagonal_mean": sums[10] / diag_count.clamp_min(1),
        "mf/t_offdiagonal_mean": sums[11] / off_count.clamp_min(1),
        "mf/gap_offdiagonal_mean": sums[12] / off_count.clamp_min(1),
        "mf/correction_rms_offdiagonal": (sums[13] / sums[14].clamp_min(1)).sqrt(),
    }
    metrics.update(zip(distribution, sums[10:]))
    return metrics


def expand_time(t, x):
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def valid_elements(x, is_pad=None):
    if is_pad is None:
        return torch.ones_like(x, dtype=torch.bool)
    if is_pad.shape != x.shape[:2]:
        raise ValueError("Action loss padding must have shape [batch, horizon]")
    return (~is_pad.bool()).unsqueeze(-1).expand_as(x)


def masked_per_sample(pred, target, is_pad=None, *, loss_type="mse"):
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    valid = valid_elements(pred, is_pad)
    count = valid.flatten(1).sum(1)
    residual = pred.float() - target.float()
    if loss_type not in {"mse", "mae"}:
        raise ValueError(f"Unknown masked loss: {loss_type}")
    error = torch.where(valid, residual.square() if loss_type == "mse" else residual.abs(), 0.0)
    return error.flatten(1).sum(1) / count.clamp_min(1), count > 0


def reduce_valid(values, valid, distributed=True):
    count = valid.sum().to(dtype=torch.float32)
    world = 1
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    # DDP averages parameter gradients; normalize by the global valid sample count.
    return torch.where(valid, values, 0.0).sum() * world / count.clamp_min(1)


def masked_mse(pred, target, is_pad=None, *, distributed=True):
    return reduce_valid(*masked_per_sample(pred, target, is_pad), distributed=distributed)


class SplitGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, t, r):
        return self.net(torch.stack((t, r), -1).to(self.net[0].weight)).squeeze(-1).float()


def sample_intervals(batch, device, shift, diagonal_probability=0.5,
                     diagonal_time_sampling="sorted_max"):
    interval_settings({"interval": dict(diagonal_probability=diagonal_probability,
                      diagonal_time_sampling=diagonal_time_sampling)}, shift)
    u = torch.rand(2, batch, device=device)
    s = shift * u / (1 + (shift - 1) * u)
    t, r = s.max(0).values, s.min(0).values
    diagonal = torch.rand(batch, device=device) < diagonal_probability
    if diagonal_time_sampling == "sorted_max":
        return t, torch.where(diagonal, t, r)
    # Independent of the interval pair AND mask. Replace BOTH endpoints.
    ud = torch.rand(batch, device=device)
    td = shift * ud / (1 + (shift - 1) * ud)
    return torch.where(diagonal, td, t), torch.where(diagonal, td, r)


@torch.no_grad()
def teacher_intervals(states, sigmas, full_probability=0.5, min_gap=1e-4):
    """States are [nodes,B,H,A], from a single descending Euler trajectory."""
    if states.ndim != 4 or sigmas.ndim != 1 or len(states) != len(sigmas):
        raise ValueError("Expected one state per trajectory node")
    if not 0 <= full_probability <= 1 or min_gap <= 0:
        raise ValueError("Invalid teacher interval configuration")
    pairs = torch.triu_indices(len(sigmas), len(sigmas), 1, device=states.device)
    gaps = sigmas[pairs[0]].float() - sigmas[pairs[1]].float()
    pairs = pairs[:, gaps >= min_gap]
    if not pairs.numel():
        raise ValueError("No teacher node pairs satisfy min_gap")
    b = states.shape[1]
    choice = torch.randint(pairs.shape[1], (b,), device=states.device)
    i, j = pairs[:, choice]
    full = torch.rand(b, device=states.device) < full_probability
    i, j = torch.where(full, 0, i), torch.where(full, len(sigmas) - 1, j)
    t, r = sigmas[i].float(), sigmas[j].float()
    gap = t - r
    if (gap < min_gap).any():
        raise ValueError("Degenerate teacher interval")
    batch = torch.arange(b, device=states.device)
    zt, zr = states[i, batch], states[j, batch]
    target = (zt.float() - zr.float()) / expand_time(gap, zt)
    return zt, r, t, target


@torch.no_grad()
def directional_derivative(fn, z, r, t, v, backend="jvp", step=0.01):
    """Hold the lower endpoint and conditioning fixed. fn internally scales time by 1000."""
    if backend == "jvp":
        _, tangent = torch.func.jvp(
            fn, (z, r, t), (v.to(z), torch.zeros_like(r), torch.ones_like(t))
        )
        return tangent.float()
    if backend != "finite_difference" or not 0 < step < 0.5:
        raise ValueError("Derivative backend must be jvp or finite_difference (0 < step < .5)")
    # Centered where legal, otherwise one-sided within r <= t <= 1.
    left = torch.minimum(torch.full_like(t, step), t - r).clamp_min(0)
    right = torch.minimum(torch.full_like(t, step), 1 - t).clamp_min(0)
    centered = (left >= step) & (right >= step)
    forward = (~centered) & (right >= left)
    hplus = torch.where(centered | forward, right, 0.0)
    hminus = torch.where(centered | ~forward, left, 0.0)
    denom = hplus + hminus
    if (denom <= 0).any():
        raise ValueError("Derivative has no legal nonzero finite-difference interval")
    a = fn((z.float() + expand_time(hplus, z) * v.float()).to(z), r, t + hplus)
    b = fn((z.float() - expand_time(hminus, z) * v.float()).to(z), r, t - hminus)
    return (a.float() - b.float()) / expand_time(denom, z)


def data_loss(
    predict,
    derivative,
    clean,
    noise,
    r,
    t,
    is_pad=None,
    *,
    objective="data_meanflow",
    generator=None,
    gamma=0.0,
    alpha=0.1,
    baseline_reduction=False,
    loss_weighting="uniform",
    adaptive_power=0.5,
    adaptive_c=0.001,
    distributed=True,
):
    """Separate live policy prediction from detached JVP values and generator graph."""
    loss_settings(dict(loss_weighting=loss_weighting,
                       adaptive_weighting=dict(power=adaptive_power, c=adaptive_c)))
    if loss_weighting == "adaptive" and objective != "data_meanflow":
        raise ValueError("Adaptive weighting is only supported for data_meanflow")
    v = noise.float() - clean.float()
    gap = t - r
    z = (clean.float() + expand_time(t, clean) * v).to(clean)
    pred = predict(z, r, t)
    active = gap > 0
    # Derivatives are evaluated only for non-diagonal rows, using a sliced condition.
    d = torch.zeros_like(v)
    gen_loss = pred.float().sum() * 0
    lam = torch.full_like(t, 0.5) if generator is None else generator(t, r)
    if objective == "data_meanflow":
        if active.any():
            d[active] = derivative(z, r, t, v, active)
        target = v - expand_time(gap, v) * d.detach()
    elif objective == "kmf":
        c = r + lam.detach() * gap
        zc = (clean.float() + expand_time(c, clean) * v).to(clean)
        d1, d2 = torch.zeros_like(v), torch.zeros_like(v)
        if active.any():
            d1[active] = derivative(z, c, t, v, active)
            d2[active] = derivative(zc, r, c, v, active)
        split = (
            expand_time((1 - lam).square(), v) * d1.detach()
            + expand_time(lam.square(), v) * d2.detach()
        )
        target = v - expand_time(gap, v) * split
        if generator is not None:
            if active.any():
                d[active] = derivative(z, r, t, v, active)
            valid = valid_elements(v, is_pad)
            norm = (
                torch.where(valid, (split - d.detach()).square(), 0.0)
                .flatten(1)
                .sum(1)
                .clamp_min(1e-24)
                .sqrt()
            )
            omega = 1 + gamma * torch.exp(-1 / (norm + 1e-6))
            residual, has_valid = masked_per_sample(pred.detach(), target, is_pad)
            gen_loss = reduce_valid(omega * residual, active & has_valid, distributed=distributed)
    else:
        raise ValueError(f"Unsupported data objective: {objective}")
    per_sample, has_valid = masked_per_sample(pred, target.detach(), is_pad)
    weights = ((per_sample.detach() + adaptive_c).pow(-adaptive_power)
               if loss_weighting == "adaptive" else torch.ones_like(per_sample))
    weighted = torch.where(has_valid, weights * per_sample, 0.0)
    policy_loss = (weighted.mean() if baseline_reduction
                   else reduce_valid(weighted, has_valid, distributed=distributed))
    effective_d = d if objective == "data_meanflow" else split.detach()
    correction = expand_time(gap, v) * effective_d
    metrics = dict(
        loss_policy=policy_loss.detach(),
        loss_generator=gen_loss.detach(),
        derivative_norm=effective_d.detach().norm(),
        correction_norm=correction.detach().norm(),
        lambda_mean=lam.detach().mean(),
        lambda_p10=torch.quantile(lam.detach(), 0.1),
        lambda_p90=torch.quantile(lam.detach(), 0.9),
    )
    if objective == "data_meanflow":
        metrics.update(loss_diagnostics(
            per_sample, weighted, weights, has_valid, gap, correction, is_pad,
            t=t, distributed=distributed,
        ))
    for label, rows in [
        ("low", t < 1 / 3),
        ("mid", (t >= 1 / 3) & (t < 2 / 3)),
        ("high", t >= 2 / 3),
    ]:
        metrics[f"correction_norm_{label}"] = correction.detach()[rows].norm()
    return policy_loss + alpha * gen_loss, metrics


def improved_data_loss(predict, derivative, predict_v, clean, noise, r, t, is_pad=None,
                       *, auxiliary=False, auxiliary_loss_weight=1.0,
                       baseline_reduction=True, loss_weighting="adaptive",
                       adaptive_power=0.5, adaptive_c=0.001, distributed=True):
    """iMF Eq.12: the JVP direction is a predicted marginal velocity, not noise-data.

    predict_v(z,t,rows) returns selected rows. Boundary is evaluated without a
    graph only off diagonal; auxiliary predicts every row once with a graph.
    Direction and JVP are detached; auxiliary supervision has its own weighting.
    """
    loss_settings(dict(loss_weighting=loss_weighting,
                       adaptive_weighting=dict(power=adaptive_power, c=adaptive_c)))
    vc = noise.float() - clean.float()
    gap = t - r
    z = (clean.float() + expand_time(t, clean) * vc).to(clean)
    pred = predict(z, r, t)
    active = gap > 0
    direction = torch.zeros_like(vc)
    vp = None
    if auxiliary:
        vp = predict_v(z, t, torch.ones_like(active))
        direction = vp.detach().float()
    elif active.any():
        with torch.no_grad():
            direction[active] = predict_v(z, t, active).detach().float()
    d = torch.zeros_like(vc)
    if active.any():
        d[active] = derivative(z, r, t, direction.detach(), active).detach()
    correction = expand_time(gap, vc) * d
    target = (vc - correction).detach()

    def weighted_loss(prediction, target):
        error, valid = masked_per_sample(prediction, target, is_pad)
        weight = ((error.detach() + adaptive_c).pow(-adaptive_power)
                  if loss_weighting == "adaptive" else torch.ones_like(error))
        weighted = torch.where(valid, weight * error, 0.0)
        loss = weighted.mean() if baseline_reduction else reduce_valid(weighted, valid, distributed)
        return loss, error, weighted, weight, valid

    lu, error, weighted, weight, valid = weighted_loss(pred, target)
    metrics = loss_diagnostics(error, weighted, weight, valid, gap, correction, is_pad,
                               t=t, distributed=distributed)
    metrics.update(loss_policy=lu.detach(), **{"imf/loss_u": lu.detach()})
    lv = lu.new_zeros(())
    if auxiliary:
        lv, ve, vw, weights_v, vv = weighted_loss(vp, vc.detach())
        vm = loss_diagnostics(ve, vw, weights_v, vv, gap, torch.zeros_like(vc), is_pad,
                              t=t, distributed=distributed)
        metrics.update({"imf/v/" + k.removeprefix("mf/"): v for k, v in vm.items()})
    # Fixed collectives on every rank, including all-diagonal/all-padding ranks.
    diagnostic_rows = valid if auxiliary else valid & active
    direction_error, _ = masked_per_sample(direction, vc, is_pad)
    mask = valid_elements(vc, is_pad) & expand_time(diagnostic_rows, vc)
    sums = torch.stack([torch.where(diagnostic_rows, direction_error, 0).sum(),
                        diagnostic_rows.sum(), torch.where(mask, direction.square(), 0).sum(),
                        mask.sum()]).float()
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums)
    metrics.update({"imf/loss_v": lv.detach(), "imf/vhat_mse": sums[0]/sums[1].clamp_min(1),
                    "imf/vhat_rms": (sums[2]/sums[3].clamp_min(1)).sqrt(),
                    "imf/vhat_samples": sums[1], "correction_norm": correction.norm(),
                    "derivative_norm": d.norm()})
    return lu + auxiliary_loss_weight * lv, metrics
