"""Real-data audit of all-flow coverage and overlapping episode task routes."""
import json
from pathlib import Path
from unittest.mock import patch

import torch

from .runtime import create_dataset, load_config
from .data import collate_flow_action


def main():
    out = Path("runs/libero_flow_implementation/all_flow")
    out.mkdir(parents=True, exist_ok=True)
    from vpp2.utils import misc
    misc.register_work_dir(out)
    cfg = load_config(["flow.missing_motion=skip",
        "flow.available_manifest=runs/libero_flow_implementation/storage_gpu_profile/available.json"])
    expected_flow = None
    reports = []
    for denominator in (8, 4, 2):
        cfg.flow.action_denominator = denominator
        cfg.flow.flow_episode_scope = "all"
        ds = create_dataset(cfg)
        flow = {(r["root"], r["episode"]):r["starts"] for r in ds.records if r["indicator"] == 1}
        action = {(r["root"], r["episode"]):r["starts"] for r in ds.records if r["indicator"] == 0}
        assert len(action) == 1712 // denominator
        missing = {tuple(x) for x in ds.audit["missing_flow_episodes"]}
        assert all((Path(root).name, eid) in missing for root,eid in set(action)-set(flow)), \
            "Missing flow on action-selected episodes must also be reported"
        if expected_flow is None: expected_flow = flow
        assert flow == expected_flow, "Flow coverage must not depend on action fraction"
        overlap = set(flow) & set(action)
        assert overlap
        key = sorted(overlap)[0]
        indices = {}
        for i,r in enumerate(ds.episodes):
            if (r["root"], r["episode"]) == key:
                indices[r["indicator"]] = ds.ends[i-1] if i else 0
        with patch.object(ds.motion_reader, "load", side_effect=AssertionError("Action route read flow")):
            a = ds[indices[0]]
        original = ds._table
        def checked(record):
            table, columns = original(record)
            assert "action" not in columns
            return table, columns
        with patch.object(ds, "_table", side_effect=checked):
            f = ds[indices[1]]
        assert "motion" not in a and "action" not in f
        assert torch.equal(a["video"], f["video"])
        batch = collate_flow_action([a,f])
        assert batch["indicator"].tolist() == [0,1]
        assert batch["action"].shape == (1,32,7)
        assert batch["motion"].shape == (1,len(ds.motion_cameras),6,8,224,224)
        assert f["motion_frame_index"].tolist() == list(range(0,32,4))
        cfg.flow.flow_episode_scope = "complementary"
        old = create_dataset(cfg)
        old_flow = {(r["root"],r["episode"]):r["starts"] for r in old.records if r["indicator"] == 1}
        assert old_flow == {k:v for k,v in flow.items() if k not in action}
        assert {(r["root"],r["episode"]):r["starts"] for r in old.records if r["indicator"] == 0} == action
        cfg.flow.action_only = True
        cfg.flow.flow_episode_scope = "all"
        only = create_dataset(cfg)
        assert only.audit["flow_episodes"] == 0
        assert only.audit["action_starts"] == ds.audit["action_starts"]
        cfg.flow.action_only = False
        row = {k:ds.audit[k] for k in ("action_fraction","action_episodes","flow_episodes",
            "action_starts","flow_starts","overlapping_action_flow_episodes","sampled_action_fraction")}
        row.update(status="passed", overlapping_episode=list(key), separate_label_routes=True,
                   legacy_complementary_compatible=True, action_only_unchanged=True)
        reports.append(row)
        print(json.dumps(row),flush=True)
    cfg.flow.missing_motion = "error"
    try:
        create_dataset(cfg)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Incomplete diagnostic manifest must not pass production coverage")
    result = dict(status="passed", fixed_manifest=str(cfg.flow.available_manifest),
        note="Fixed diagnostic coverage only; production requires complete flow GT for every episode.",
        flow_set_and_starts_identical_across_fractions=True, missing_gt_rejected=True, variants=reports)
    (out/"validation.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
