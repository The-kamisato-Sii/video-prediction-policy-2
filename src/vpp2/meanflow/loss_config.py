"""Loss contract shared by configuration and checkpoint compatibility checks."""

import math


def loss_settings(config):
    # Missing fields in historical configurations always mean uniform.
    mode = config.get("loss_weighting", "uniform")
    adaptive = config.get("adaptive_weighting", {})
    power = float(adaptive.get("power", 0.5))
    c = float(adaptive.get("c", 0.001))
    if mode not in {"uniform", "adaptive"}:
        raise ValueError(f"Unknown MeanFlow loss_weighting: {mode}")
    if not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("adaptive_weighting.power must be finite and in [0, 1]")
    if not math.isfinite(c) or c <= 0:
        raise ValueError("adaptive_weighting.c must be finite and positive")
    return dict(loss_weighting=mode, adaptive_weighting=dict(power=power, c=c))


def validate_loss_resume(current, saved):
    a, b = loss_settings(current), loss_settings(saved)
    if a["loss_weighting"] != b["loss_weighting"] or (
        a["loss_weighting"] == "adaptive" and a != b
    ):
        raise ValueError(f"MeanFlow loss settings differ on resume: current={a}, saved={b}")
