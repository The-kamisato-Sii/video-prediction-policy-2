"""Single-GPU capacity measurements with real data and full optimizer state.

Run directly for plain AdamW, or with the isolated Accelerate/ZeRO1 launcher.
These short optimizer updates are diagnostics, never production training.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from accelerate.utils import set_seed

from .data import collate_flow_action
from .runtime import load_config, create_dataset, create_model
from .trainer import LiberoFlowTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--tokens", type=int, choices=[32, 64, 128], default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    from vpp2.utils import misc
    from vpp2.utils.logging_config import setup_logging
    setup_logging()
    misc.register_work_dir(out)
    cfg = load_config([f"output_dir={out}", f"flow.tokens={args.tokens}",
        "flow.missing_motion=skip", "batch_size=1", "gradient_accumulation_steps=1",
        "max_steps=1", "num_workers=0", "eval_every=0", "wandb.enabled=false",
        "save_final_checkpoint=false"])
    set_seed(int(cfg.seed))
    dataset = create_dataset(cfg)
    samples = {mode: [dataset[dataset.first_index(mode, suite)]
        for suite in ("spatial", "object", "goal", "libero_10")] for mode in (0, 1)}
    print("Loading full pretrained model for single-GPU capacity measurement", flush=True)
    model = create_model(cfg)
    trainer = LiberoFlowTrainer(model, dataset, cfg=cfg)
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Capacity probe must own exactly one GPU/process")
    trainer._set_dit_only_train_mode()
    report = dict(status="running", gpu=torch.cuda.get_device_name(), tokens=args.tokens,
        total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        backend=str(trainer.accelerator.distributed_type), zero_stage=trainer._deepspeed_zero_stage(),
        note="Diagnostic updates with variable input batch size; no production checkpoints. "
             "ZeRO1 uses one rank, so optimizer state is not partitioned across two GPUs.",
        trials=[], largest_complete_batch=0, recommended_batch_with_10pct_headroom=0)

    def save():
        (out / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")

    def run_case(batch_size, mode):
        if mode == "action":
            labels = [0] * batch_size
        elif mode == "flow":
            labels = [1] * batch_size
        else:
            n_action = max(1, batch_size // 8)
            labels = [0] * n_action + [1] * (batch_size - n_action)
        batch = collate_flow_action([samples[label][i % 4] for i, label in enumerate(labels)])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        for _ in range(2):
            with trainer.accelerator.accumulate(trainer.model):
                with trainer.accelerator.autocast():
                    loss, metrics = trainer.model(batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss: {metrics}")
                trainer.accelerator.backward(loss)
                trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad(set_to_none=True)
            del loss
        torch.cuda.synchronize()
        return dict(batch_size=batch_size, labels=mode, status="passed", updates=2,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            seconds=time.monotonic()-start, metrics=metrics)

    save()
    for batch_size in sorted(set(args.batches)):
        peaks = []
        for mode in ("action", "flow", "mixed"):
            if batch_size == 1 and mode == "mixed":
                continue
            print(f"Capacity trial: N={args.tokens} batch={batch_size} labels={mode}", flush=True)
            try:
                row = run_case(batch_size, mode)
            except torch.cuda.OutOfMemoryError as error:
                report["trials"].append(dict(batch_size=batch_size, labels=mode,
                    status="out_of_memory", error=str(error)[:1500]))
                report["status"] = "capacity_boundary_measured"
                save()
                print(json.dumps(report), flush=True)
                return
            report["trials"].append(row)
            peaks.append(row["peak_reserved_gib"])
            save()
            print(json.dumps(row), flush=True)
        report["largest_complete_batch"] = batch_size
        if max(peaks) < report["total_gib"] * 0.9:
            report["recommended_batch_with_10pct_headroom"] = batch_size
        save()
    report["status"] = "all_requested_batches_passed"
    save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
