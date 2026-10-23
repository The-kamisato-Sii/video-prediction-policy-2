"""Shared ablation contracts for launch, training, deployment and resume."""

import math
from pathlib import Path


def interval_settings(config, train_shift=None):
    raw = config.get("interval", {})
    mode = raw.get("diagonal_time_sampling", "sorted_max")
    probability = float(raw.get("diagonal_probability", 0.5))
    if mode not in {"sorted_max", "independent"}:
        raise ValueError(f"Unknown diagonal_time_sampling: {mode}")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("diagonal_probability must be finite and in [0, 1]")
    result = dict(raw, source=raw.get("source", "shifted_uniform"),
                  diagonal_probability=probability, diagonal_time_sampling=mode)
    shift = train_shift if train_shift is not None else raw.get("train_shift")
    if shift is not None:
        shift = float(shift)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("train shift must be finite and positive")
        if train_shift is not None and "train_shift" in raw and float(raw["train_shift"]) != shift:
            raise ValueError("Recorded interval train_shift conflicts with action scheduler")
        result["train_shift"] = shift  # Recorded scheduler value, never a separate knob.
    return result


def validate_interval_objective(config, benchmark):
    interval = interval_settings(config)
    if interval["diagonal_time_sampling"] == "independent" and not (
        benchmark == "libero" and config.get("objective") == "data_meanflow"
        and interval["source"] == "shifted_uniform"
    ):
        raise ValueError("independent diagonal sampling requires LIBERO data_meanflow shifted_uniform")
    if config.get("objective") in {"data_meanflow", "kmf"} and interval["source"] != "shifted_uniform":
        raise ValueError("Data interval sampling requires source=shifted_uniform")


def direct_action_settings(config):
    raw = config.get("direct_action", {})
    loss, inputs = raw.get("loss", "mse"), raw.get("input", "zero")
    if loss not in {"mse", "mae"} or inputs not in {"zero", "learnable"}:
        raise ValueError("Direct Action requires loss=mse|mae and input=zero|learnable")
    result = dict(loss=loss, input=inputs)
    if inputs == "learnable":
        length = raw.get("query_length", 32)
        if isinstance(length, bool) or int(length) != length or int(length) <= 0:
            raise ValueError("query_length must be a positive integer")
        result["query_length"] = int(length)
    return result


def _historical_shift(resolved_config, checkpoint_path):
    """Recover actual scheduler settings from saved evidence, never the new run."""
    from omegaconf import OmegaConf

    candidates = [resolved_config] if resolved_config else []
    if checkpoint_path:
        for parent in list(Path(checkpoint_path).resolve().parents)[:4]:
            path = parent / "config.yaml"
            if path.is_file():
                candidates.append(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
                break
    for cfg in candidates:
        shift = cfg.get("model", {}).get("action_scheduler", {}).get("train_shift")
        if shift is not None:
            return float(shift)
    return None


def validate_experiment_resume(current, saved, *, resolved_config=None, checkpoint_path=None):
    objective = current.get("objective")
    if objective != saved.get("objective"):
        raise ValueError("Resume objective differs; start a separate experiment")
    if objective == "direct_action":
        a, b = direct_action_settings(current), direct_action_settings(saved)
    elif objective in {"data_meanflow", "kmf"}:
        a, b = interval_settings(current), interval_settings(saved)
        if "train_shift" in a and "train_shift" not in b:
            shift = _historical_shift(resolved_config, checkpoint_path)
            if shift is None:
                raise ValueError("Cannot verify historical action train_shift for resume; saved config is required")
            b = interval_settings(saved, shift)
        keys = ("source", "diagonal_probability", "diagonal_time_sampling", "train_shift")
        a, b = ({key: settings.get(key) for key in keys} for settings in (a, b))
    else:
        return
    if a != b:
        raise ValueError(f"Experiment settings differ on resume: current={a}, saved={b}")
