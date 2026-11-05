"""Automatic GPU topology and effective global batch256, with memory-aware microbatches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import re


def batch_plan(world, memory_gib, micro_override=None):
    if world < 1 or 256 % world:
        raise ValueError("GPU world size must divide256")
    effective = 256 // world
    cap = 16 if memory_gib >= 120 else 4 if memory_gib >= 75 else 1
    micro = min(effective, cap) if micro_override is None else int(micro_override)
    if micro < 1 or effective % micro:
        raise ValueError("Per-GPU microbatch must divide256/world")
    return dict(global_batch=256, world_size=world, effective_batch_per_gpu=effective,
                micro_batch=micro, gradient_accumulation_steps=effective//micro)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, choices=(32,64,128), required=True)
    p.add_argument("--action-denominator", type=int, choices=(1,2,4,8), required=True)
    p.add_argument("--flow-dt", type=int, choices=(1,4), default=4)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--objective", choices=("latent_flow_matching", "direct_reconstruction"), default="latent_flow_matching")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("overrides", nargs="*")
    args=p.parse_args()
    for item in args.overrides:
        if item.startswith(("batch_size=", "gradient_accumulation_steps=", "reference_global_batch_size=")):
            raise ValueError("Use --micro-batch; effective global batch remains256")
        if item.startswith("flow.objective="):
            raise ValueError("Use --objective so output directories distinguish the ablation")
    gpus=torch.cuda.device_count()
    nproc=int(os.environ.get("NPROC_PER_NODE",os.environ.get("PET_NPROC_PER_NODE",gpus)))
    nodes=int(os.environ.get("NNODES",os.environ.get("PET_NNODES",1)))
    rank=int(os.environ.get("NODE_RANK",os.environ.get("PET_NODE_RANK",0)))
    if nproc < 1 or nproc > gpus:
        raise ValueError(f"Requested{nproc} GPUs, visible{gpus}")
    memory=min(torch.cuda.get_device_properties(i).total_memory/2**30 for i in range(nproc))
    plan=batch_plan(nodes*nproc,memory,args.micro_batch)
    prefix = "recon_" if args.objective == "direct_reconstruction" else ""
    from .runtime import load_config
    flow_settings = load_config(args.overrides).flow
    flow_scope = flow_settings.flow_episode_scope
    if list(flow_settings.motion_cameras) == ["image"]:
        prefix += "headflow_"
    if flow_scope == "all":
        prefix += "allflow_"
    name=os.environ.get("RUN_NAME",f"{prefix}n{args.tokens}_a1of{args.action_denominator}_dt{args.flow_dt}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name):
        raise ValueError("RUN_NAME must be a simple directory name")
    out=Path("runs/libero_flow")/name
    cfg=Path("scripts/libero_flow/accelerate_zero1.yaml")
    overrides=[f"flow.tokens={args.tokens}",f"flow.action_denominator={args.action_denominator}",
               f"flow.objective={args.objective}",
               f"flow.dt={args.flow_dt}",f"output_dir={out}",
               f"batch_size={plan['micro_batch']}",f"gradient_accumulation_steps={plan['gradient_accumulation_steps']}",*args.overrides]
    if not args.dry_run:
        # Replay generation can finish episodes while workers initialize. Freeze
        # coverage once for production AND diagnostics, BEFORE spawning ranks,
        # and reuse on resume so sample indices stay identical across ranks.
        from .runtime import load_config
        cfg_data=load_config(overrides)
        if cfg_data.flow.missing_motion == "skip" and not cfg_data.flow.action_only and not cfg_data.flow.get("available_manifest"):
            manifest=Path("runs/libero_flow_manifests") / f"{name}.json"
            if rank == 0 and not manifest.exists():
                print("Collecting completed motion episodes for a fixed flow manifest", flush=True)
                episodes=[]
                for directory in cfg_data.data.train.dataset_dirs:
                    root=Path(directory)
                    for path in sorted((root/"meta/motions").glob("episode_*.json")):
                        if path.stem.endswith("_validation"): continue
                        meta=json.loads(path.read_text())
                        eid=meta.get("episode")
                        if meta.get("complete_episode") and meta.get("replay_version")==3 and eid is not None:
                            motion=root/f"motions/chunk-{eid//1000:03d}/episode_{eid:06d}.hdf5"
                            if motion.is_file(): episodes.append([root.name,int(eid)])
                manifest.parent.mkdir(parents=True,exist_ok=True)
                temporary=manifest.with_suffix(".tmp")
                temporary.write_text(json.dumps({"episodes":episodes},indent=2)+'\n')
                temporary.replace(manifest)
            if not manifest.exists():
                raise FileNotFoundError("Start node0 first so the shared diagnostic manifest exists")
            overrides.append(f"flow.available_manifest={manifest.resolve()}")
    cmd=[sys.executable,"-m","accelerate.commands.launch","--config_file",str(cfg),
         "--num_machines",str(nodes),"--num_processes",str(nodes*nproc),"--machine_rank",str(rank),
         "--main_process_port",os.environ.get("MASTER_PORT",os.environ.get("PET_MASTER_PORT","29637"))]
    addr=os.environ.get("MASTER_ADDR",os.environ.get("PET_MASTER_ADDR"))
    if nodes>1 and not addr:
        raise ValueError("Multi-node launch needs MASTER_ADDR/PET_MASTER_ADDR")
    if addr:
        cmd += ["--main_process_ip",addr]
    cmd += ["--module","vpp2.libero_flow.runtime"]
    if args.diagnostic: cmd += ["--diagnostic"]
    cmd += overrides
    print(json.dumps(dict(batch=plan,memory_gib=memory,command=cmd),indent=2),flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd,check=True)


if __name__ == "__main__":
    main()
