"""Actual-data label audit and actual pretrained single-GPU shape/gradient checks."""
import argparse
import gc
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .runtime import load_config, create_dataset, create_model
from .data import collate_flow_action
from .model import FlowActionAdapters


def dump(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    summary={k:v for k,v in report.items() if k not in ("checks", "baseline_comparison", "missing_flow_episodes")}
    if "missing_flow_episodes" in report:
        summary["missing_flow_episodes"]=len(report["missing_flow_episodes"])
        summary["baseline_compared_samples"]=len(report.get("baseline_comparison", []))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def validate_data(cfg, out, compare_baseline=True):
    from vpp2.utils import misc
    misc.register_work_dir(out)
    reports=[]
    for denominator in (8,4,2):
        cfg.flow.action_denominator=denominator
        ds=create_dataset(cfg)
        report=dict(ds.audit)
        report["checks"]=[]
        for mode in (0,1):
            for suite in ("spatial", "object", "goal", "libero_10"):
                index=ds.first_index(mode,suite)
                r,start=ds.locate(index)
                # Fail loudly if the forbidden label source is accessed.
                if mode==0:
                    with patch.object(ds.motion_reader, "load", side_effect=AssertionError("Action sample opened motion GT")):
                        sample=ds[index]
                    assert "motion" not in sample
                else:
                    ds._tables.clear()
                    import pyarrow.parquet as pq
                    original=pq.read_table
                    def checked(path, *args, **kwargs):
                        assert "action" not in kwargs["columns"]
                        return original(path,*args,**kwargs)
                    with patch("vpp2.libero_flow.data.pq.read_table", side_effect=checked):
                        sample=ds[index]
                    assert "action" not in sample
                    assert sample["motion"].shape==(len(ds.motion_cameras),6,8,224,224)
                    assert sample["motion_frame_index"].tolist()==[0,4,8,12,16,20,24,28]
                    assert torch.isfinite(sample["motion"]).all()
                    assert sample["motion_valid"][:,1,-1].any()
                report["checks"].append(dict(indicator=mode,suite=suite,provenance=sample["provenance"],
                    shapes={k:list(v.shape) for k,v in sample.items() if torch.is_tensor(v)}))
        a=ds[ds.first_index(0)];f=ds[ds.first_index(1)]
        mixed=collate_flow_action([a,f])
        assert mixed["indicator"].tolist()==[0,1]
        assert mixed["action"].shape==(1,32,7) and mixed["motion"].shape==(1,len(ds.motion_cameras),6,8,224,224)
        report["missing_labels_not_loaded"]=True
        if compare_baseline:
            baseline_cfg=OmegaConf.create(OmegaConf.to_container(cfg.data.train,resolve=True))
            baseline_cfg.episode_manifest=f"configs/libero/subsets/episodes_1of{denominator}_seed42.json"
            baseline=instantiate(baseline_cfg)
            multi=baseline.lerobot_dataset.multi_dataset
            comparisons=[]
            offset=0
            for sub in multi._datasets:
                # The reference loader is restricted to A episodes only.
                episode_ids=np.asarray(sub.hf_dataset["episode_index"])
                root=Path(sub.root)
                own=next((i for i,r in enumerate(ds.episodes) if Path(r["root"]).name==root.name and r["indicator"]==0),None)
                if own is None:
                    raise AssertionError(root)
                record=ds.episodes[own]
                source_start=ds.ends[own-1] if own else 0
                positions=np.flatnonzero(episode_ids==record["episode"])
                for local in (0,record["length"]//2,record["length"]-1):
                    ref=baseline._get(int(offset+positions[local]),strict_index=True)
                    new=ds[source_start+local]
                    errors={k:float((ref[k].float()-new[k].float()).abs().max()) for k in ("action","video","proprio","context")}
                    assert all(v<=1e-6 for v in errors.values()),errors
                    assert torch.equal(ref["action_is_pad"],new["action_is_pad"])
                    comparisons.append(dict(suite=root.name,episode=record["episode"],start=local,max_errors=errors))
                offset+=len(sub)
            report["baseline_comparison"]=comparisons
            del baseline
        reports.append(report)
        dump(out/f"data_1of{denominator}.json",report)
        del ds
        gc.collect()
    return reports


def validate_model(cfg, out, tokens):
    cfg.flow.action_denominator=8
    ds=create_dataset(cfg)
    samples={mode:ds[ds.first_index(mode)] for mode in (0,1)}
    print("Loading the full baseline Wan/ActionDiT and official MotionVAE weights",flush=True)
    model=create_model(cfg)
    print("Full model loaded",flush=True)
    reports=[]
    for n in tokens:
        torch.manual_seed(int(cfg.seed))
        model.flow_adapters=FlowActionAdapters(model.action_expert.hidden_dim,n,motion_channels=model.flow_codec.latent_channels).to(model.device,model.torch_dtype)
        model.configure_trainability()
        model.last_shapes={}
        torch.cuda.reset_peak_memory_stats()
        report=dict(tokens=n,branches=[])
        for mode in (0,1):
            batch=collate_flow_action([samples[mode]])
            print(f"Validating N={n}, indicator={mode}: forward/backward then10 sampler evaluations",flush=True)
            model.zero_grad(set_to_none=True)
            loss,metrics=model(batch)
            assert torch.isfinite(loss)
            loss.backward()
            p=model.action_expert.blocks[0].ffn[0].weight
            grad=float(p.grad.float().norm()) if p.grad is not None else 0.0
            indicator_grad=float(model.flow_adapters.indicator.weight.grad[mode].float().norm())
            assert grad>0 and indicator_grad>0,(grad,indicator_grad)
            assert not any(p.grad is not None for p in model.video_expert.parameters())
            assert not any(p.grad is not None for p in model.flow_codec.parameters())
            row=dict(indicator=mode,loss=float(loss.detach()),metrics=metrics,
                     backbone_grad_norm=grad,indicator_grad_norm=indicator_grad,
                     shapes=dict(model.last_shapes[str(mode)]))
            model.zero_grad(set_to_none=True)
            del loss
            gc.collect()
            model.eval()
            with torch.no_grad():
                prediction=model.sample_task(batch,indicator=mode,steps=10)
            assert all(torch.isfinite(x).all() for x in prediction.values())
            row["sampled_shapes"]={k:list(v.shape) for k,v in prediction.items()}
            if mode==0:
                assert prediction["action"].shape==(1,32,7)
            else:
                assert prediction["scene_flow"].shape==(1,len(ds.motion_cameras),3,8,224,224)
                row["codec_roundtrip"]={}
                with torch.no_grad():
                    maps=batch["motion"].to(model.device,model.torch_dtype)
                    z=model.flow_codec.encode(maps)
                    recon=model.flow_codec.decode(z)
                    row["codec_roundtrip"]=dict(latent_shape=list(z.shape),decoded_shape=list(recon.shape),
                        rmse=float((recon.float()-maps.float()).square().mean().sqrt()))
            report["branches"].append(row)
            print(json.dumps(row),flush=True)
            del prediction
            model.configure_trainability()
        report["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
        report["status"]="passed"
        dump(out/f"model_n{n}.json",report)
        reports.append(report)
    return reports


def validate_temporal(cfg,out):
    ds=create_dataset(cfg)
    rows=[]
    for suite in ("spatial","object","goal","libero_10"):
        idx=ds.first_index(1,suite)
        ds.flow_dt=1;one=ds[idx]
        ds.flow_dt=4;four=ds[idx]
        assert torch.equal(one["motion_frame_index"],four["motion_frame_index"])
        assert torch.equal(one["motion"][:,:3],four["motion"][:,:3])
        assert torch.equal(one["motion_center"],four["motion_center"])
        assert torch.equal(one["motion_scale"],four["motion_scale"])
        for dt,item in ((1,one),(4,four)):
            assert torch.equal(item["flow_target_frame_index"],item["motion_frame_index"]+dt)
            assert item["motion_valid"][:,1,-1].any()
        rows.append(dict(suite=suite,sources=one["motion_frame_index"].tolist(),
                         dt1_endpoints=one["flow_target_frame_index"].tolist(),
                         dt4_endpoints=four["flow_target_frame_index"].tolist(),
                         identical_geometry_and_normalization=True,last_flow_valid=True))
    dump(out/"temporal_dt1_dt4.json",dict(status="passed",suites=rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",action="store_true")
    p.add_argument("--model",action="store_true")
    p.add_argument("--temporal",action="store_true")
    p.add_argument("--skip-baseline-comparison",action="store_true")
    p.add_argument("--tokens",type=int,nargs="+",default=[32,64,128])
    p.add_argument("--allow-incomplete-gt",action="store_true")
    p.add_argument("--output",default="runs/libero_flow_implementation/validation")
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(["wandb.enabled=false"])
    if args.allow_incomplete_gt: cfg.flow.missing_motion="skip"
    if args.data: validate_data(cfg,out,not args.skip_baseline_comparison)
    if args.model: validate_model(cfg,out,args.tokens)
    if args.temporal: validate_temporal(cfg,out)


if __name__=="__main__":
    main()
