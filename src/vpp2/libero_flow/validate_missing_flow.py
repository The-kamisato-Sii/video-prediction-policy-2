"""Verify production skip policy using all actual LIBERO episode metadata."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from . import runtime, launch
from .data import collate_flow_action


def main():
    out = Path("runs/libero_flow_implementation/skip_missing_flow_20261025")
    out.mkdir(parents=True, exist_ok=True)
    from vpp2.utils import misc
    misc.register_work_dir(out)
    cfg = runtime.load_config()
    assert cfg.flow.missing_motion == "skip"
    assert cfg.flow.flow_episode_scope == "all"
    # Exercise the real production launcher, intercepting only GPU discovery
    # and process execution. It must freeze a shared manifest without --diagnostic.
    name = "validate-skip-flow-20261025"
    with patch.dict(os.environ, {"RUN_NAME":name,"NPROC_PER_NODE":"2","NNODES":"1","NODE_RANK":"0"}), \
         patch("sys.argv", ["launch","--tokens","32","--action-denominator","8"]), \
         patch.object(torch.cuda,"device_count",return_value=2), \
         patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(total_memory=140*2**30)), \
         patch.object(launch.subprocess,"run") as run:
        launch.main()
        command = run.call_args.args[0]
        assert "--diagnostic" not in command
        manifest_arg = next(x for x in command if x.startswith("flow.available_manifest="))
    cfg.flow.available_manifest = manifest_arg.split("=",1)[1]
    manifest = json.loads(Path(cfg.flow.available_manifest).read_text())
    expected = {tuple(x) for x in manifest["episodes"]}
    rows=[]
    flow_starts=None
    for denominator in (8,4,2):
        cfg.flow.action_denominator=denominator
        ds=runtime.create_dataset(cfg)
        a={(Path(r["root"]).name,r["episode"]) for r in ds.records if r["indicator"]==0}
        f={(Path(r["root"]).name,r["episode"]) for r in ds.records if r["indicator"]==1}
        selected=json.loads(Path(f"configs/libero/subsets/episodes_1of{denominator}_seed42.json").read_text())
        expected_a={(Path(k).name,eid) for k,ids in selected["episodes_by_dataset"].items() for eid in ids}
        assert a==expected_a and len(a)==1712//denominator
        assert f==expected
        if flow_starts is None:flow_starts=ds.audit["flow_starts"]
        assert ds.audit["flow_starts"]==flow_starts
        missing={tuple(x) for x in ds.audit["missing_flow_episodes"]}
        assert (a-f)<=missing
        assert len(a-f)==ds.audit["action_episodes_without_flow"]
        # An action episode rejected for flow must still yield a real action
        # training sample, without attempting to read any motion target.
        key=sorted(a-f)[0]
        for i,r in enumerate(ds.episodes):
            if r["indicator"]==0 and (Path(r["root"]).name,r["episode"])==key:
                index=ds.ends[i-1] if i else 0
                break
        with patch.object(ds.motion_reader,"load",side_effect=AssertionError("Action read missing flow")):
            sample=ds[index]
        batch=collate_flow_action([sample])
        assert batch["action"].shape==(1,32,7)
        assert "motion" not in batch and batch["flow_indices"].numel()==0
        row={k:ds.audit[k] for k in ("action_fraction","action_episodes","flow_episodes",
            "action_starts","flow_starts","skipped_flow_episodes","action_episodes_without_flow")}
        row["action_without_flow_sample"]=list(key)
        rows.append(row)
        print(json.dumps(row),flush=True)
    # Enter the production runtime's validation path, without starting a model.
    with patch.object(runtime,"train") as train:
        runtime.main(["batch_size=1","gradient_accumulation_steps=256"])
        assert train.call_count==1
        assert train.call_args.args[0].flow.missing_motion=="skip"
    result=dict(status="passed",production_skip_accepted=True,
        production_manifest_frozen=True,manifest=cfg.flow.available_manifest,
        original_action_manifests_exactly_preserved=True,variants=rows)
    (out/"validation.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
