"""Prepare/submit a two-rank H200 diagnostic through the supplied qzcli tool.

The caller must supply a valid platform specification. Allocating8 GPUs instead
of the requested2 requires separate user authorization and an explicit flag.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--submit",action="store_true")
    p.add_argument("--name")
    p.add_argument("--spec-id",default="")
    p.add_argument("--allocation-gpus",type=int,choices=(2,8),default=2,
                   help="Use8 only after explicit authorization to allocate8 GPUs")
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    project=repo.parent
    out=repo/"runs/libero_flow_implementation"
    out.mkdir(parents=True,exist_ok=True)
    group="lcg-875f1ead-413f-4756-a8f6-45d319402490"
    ws="ws-a2836af9-fddd-40cc-b5b2-3d6df79d651a"
    name=args.name or f"vpp2-flow-2rank-{args.allocation_gpus}h200-validation-1023"
    stem="two_h200" if args.allocation_gpus==2 else "two_h200_on_eight_gpu_allocation"
    # The user's actual2-H200 notebook specification is40 CPU /400 GiB RAM.
    cpu = args.allocation_gpus * 20
    mem = 400 if args.allocation_gpus == 2 else 1800
    body="set -euo pipefail\ncd "+shlex.quote(str(repo))+"\nexport CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2\nexec bash scripts/libero_flow/validate_distributed.sh\n"
    payload=dict(name=name,workspace_id=ws,
        project_id="project-4aeee1e2-1960-487e-a492-9c70c96fdf77",project_en_name="embodied-model",
        logic_compute_group_id=group,framework="pytorch",task_priority=10,
        command="bash -c "+shlex.quote(body),dataset_info=[],envs=[],mounts=[],
        enable_notification=False,auto_fault_tolerance=False,
        framework_config=[dict(instance_count=1,gpu_count=args.allocation_gpus,cpu=cpu,mem_gi=mem,shm_gi=256,
            image="docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3",image_type="SOURCE_OFFICIAL",
            resource_spec_price=dict(cpu_type="",cpu_count=cpu,gpu_type="NVIDIA_H200_SXM_141G",
                gpu_count=args.allocation_gpus,memory_size_gib=mem,logic_compute_group_id=group,quota_id=args.spec_id))])
    (out/f"{stem}_payload.json").write_text(json.dumps(payload,indent=2)+'\n')
    if not args.submit:
        print(json.dumps(payload,indent=2));return
    if not args.spec_id:
        raise ValueError("Platform requires a valid quota_id; pass --spec-id for the requested allocation")
    for n in (32,64,128):
        report=json.loads((out/f"validation/model_n{n}.json").read_text())
        assert report["status"]=="passed" and len(report["branches"])==2
    for d in (8,4,2):
        report=json.loads((out/f"validation/data_1of{d}.json").read_text())
        assert report["missing_labels_not_loaded"] and len(report["baseline_comparison"])==12
    os.environ.setdefault("QZCLI_HOME",str(project/"tools/qzcli-state"))
    no_proxy=os.environ.get("NO_PROXY","")+",qz.sii.edu.cn,.sii.edu.cn"
    os.environ["NO_PROXY"]=os.environ["no_proxy"]=no_proxy
    sys.path.insert(0,str(project/"tools/qzcli_tool"))
    from qzcli.api import QzAPI
    from qzcli.config import get_cookie
    api=QzAPI();cookie=get_cookie()["cookie"]
    jobs=api.list_jobs_with_cookie(ws,cookie,page_size=200)["jobs"]
    matches=[j for j in jobs if j["name"]==name]
    if len(matches)>1: raise RuntimeError("Duplicate job names; inspect before submitting")
    try:
        result={"job_id":matches[0]["job_id"]} if matches else api.create_job_v2(cookie,payload)
    except Exception as error:
        (out/f"{stem}_rejection.json").write_text(json.dumps(dict(error=str(error),requested_gpus=args.allocation_gpus),indent=2)+'\n')
        raise
    (out/f"{stem}_submission.json").write_text(json.dumps(result,indent=2)+'\n')
    detail=api.get_job_detail(result["job_id"])
    (out/f"{stem}_detail.json").write_text(json.dumps(detail,indent=2)+'\n')
    count=sum(f["gpu_count"]*f["instance_count"] for f in detail["framework_config"])
    if count!=args.allocation_gpus: raise RuntimeError(f"Unexpected allocation request count{count}; inspect immediately")
    print(json.dumps(dict(job_id=result["job_id"],status=detail["status"],requested_gpus=count),indent=2))


if __name__=="__main__":
    main()
