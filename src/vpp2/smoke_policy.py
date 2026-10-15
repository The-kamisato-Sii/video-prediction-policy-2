"""Load released weights and exercise the complete RoboDojo observation-to-action path.

Synthetic mode checks executability only; it is not a simulator evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vpp2.policy import RoboDojoVPP2Policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--wan", required=True)
    obs = p.add_mutually_exclusive_group(required=True)
    obs.add_argument(
        "--observation", help="NPZ with head,left,right,state,instruction (allow_pickle=False)"
    )
    obs.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--requests", type=int, default=3)
    p.add_argument("--text-encoder-device", choices=["cpu"])
    p.add_argument("--offload-clip", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.requests < 1 or args.warmup < 0:
        raise ValueError("Invalid request counts")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.synthetic:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        views = {k: image.copy() for k in ("head", "left", "right")}
        state = np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5] * 2, dtype=np.float32)
        instruction = "pick up the block"
    else:
        with np.load(args.observation, allow_pickle=False) as data:
            views = {k: data[k].copy() for k in ("head", "left", "right")}
            state = data["state"].copy()
            instruction = str(data["instruction"].item())
    observation = {
        "observation": {
            camera: {"rgb": views[key]}
            for camera, key in [
                ("head_camera", "head"),
                ("left_camera", "left"),
                ("right_camera", "right"),
            ]
        },
        "joint_action": {"vector": state},
        "env_idx": 0,
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    policy = RoboDojoVPP2Policy(
        Path(args.bundle) / "action.pt",
        args.wan,
        num_inference_steps=args.steps,
        text_encoder_device=args.text_encoder_device,
        offload_clip=args.offload_clip,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    counters = {"action": 0, "prefill": 0}
    layers = [0] * len(policy.model.action_expert.blocks)
    original = policy.model.mot.forward_action_with_video_cache
    prefill = policy.model.mot.prefill_video_cache

    def count_action(*a, **kw):
        counters["action"] += 1
        return original(*a, **kw)

    def count_prefill(*a, **kw):
        counters["prefill"] += 1
        return prefill(*a, **kw)

    policy.model.mot.forward_action_with_video_cache = count_action
    policy.model.mot.prefill_video_cache = count_prefill
    handles = []
    for index, block in enumerate(policy.model.action_expert.blocks):

        def hook(module, args, out, index=index):
            layers[index] += 1

        handles.append(block.self_attn.q.register_forward_hook(hook))
    stage_seconds = {}

    def instrument(owner, method, label):
        original_method = getattr(owner, method)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_method(*args, **kwargs)
            torch.cuda.synchronize()
            stage_seconds[label] = stage_seconds.get(label, 0.0) + time.perf_counter() - started
            return result

        setattr(owner, method, timed)

    for method, label in [
        ("_compose_observation_image", "rgb_preprocess"),
        ("_build_history_condition", "history"),
        ("_normalize_state", "state_normalization"),
        ("_denormalize_action", "action_postprocess"),
    ]:
        instrument(policy, method, label)
    for method, label in [
        ("_encode_video_latents", "vae"),
        ("_encode_clip_fea_from_video", "clip"),
        ("encode_prompt", "text_encoder"),
        ("_predict_action_noise_with_cache", "action"),
    ]:
        instrument(policy.model, method, label)
    instrument(policy.model.mot, "prefill_video_cache", "video_prefill")
    timings = []
    records = []
    stages = []
    for i in range(args.warmup + args.requests):
        policy.reset()
        stage_seconds.clear()
        counters.update(action=0, prefill=0)
        layers[:] = [0] * len(layers)
        torch.cuda.synchronize()
        start = time.perf_counter()
        policy.observe(observation)
        actions = policy._infer_action_chunk(observation, instruction)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if actions.shape != (32, 16) or not np.isfinite(actions).all():
            raise RuntimeError("Invalid action chunk")
        if counters != {"action": args.steps, "prefill": 1} or layers != [args.steps] * 40:
            raise RuntimeError(f"Unexpected actual calls: {counters}, {layers}")
        if i >= args.warmup:
            timings.append(elapsed)
            stages.append(dict(stage_seconds))
            records.append(dict(counters, action_layer_calls=list(layers)))
            np.save(output / f"actions_{i - args.warmup:03d}.npy", actions)
    for h in handles:
        h.remove()
    report = dict(
        kind="synthetic_policy_smoke" if args.synthetic else "recorded_observation_policy_smoke",
        closed_loop=False,
        checkpoint=str(Path(args.bundle).resolve()),
        action_sampler=policy.model.action_sampler,
        steps=args.steps,
        shift=1,
        seed=1,
        text_encoder_device=args.text_encoder_device or "cuda",
        offload_clip=args.offload_clip,
        load_seconds=load_seconds,
        seconds=timings,
        p50_seconds=float(np.percentile(timings, 50)),
        p95_seconds=float(np.percentile(timings, 95)),
        calls=records,
        stage_seconds=stages,
        timing_note="CUDA-synchronized instrumented methods; end-to-end includes instrumentation overhead",
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
