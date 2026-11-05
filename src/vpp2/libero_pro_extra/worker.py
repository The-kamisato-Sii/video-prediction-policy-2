import argparse
from pathlib import Path
from omegaconf import OmegaConf
from .runtime import register_environment_suites

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=OmegaConf.load(a.config)
    register_environment_suites()
    from vpp2.libero.worker import check_runtime
    tasks=[(x.split(',')[0],int(x.split(',')[1])) for x in Path(cfg.EVALUATION.task_manifest).read_text().splitlines() if x.strip()]
    check_runtime(cfg,tasks)
    from vpp2.libero.eval_libero_task_manifest import eval_task_manifest
    eval_task_manifest(cfg)
if __name__=='__main__':main()
