"""Profile a real PRO worker; a phase barrier keeps loading out of rollout timing."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
from omegaconf import OmegaConf


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--profile-dir',required=True)
    args=p.parse_args()
    root=Path(args.profile_dir);root.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(args.config)
    started=time.monotonic()
    report=dict(gpu_id=int(cfg.gpu_id),pid=os.getpid(),tasks=[],status='RUNNING')
    def save():
        tmp=root/'profile.json.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(root/'profile.json')
    from vpp2.libero.worker import check_runtime
    tasks=[(s,int(t)) for s,t in (line.split(',') for line in Path(cfg.EVALUATION.task_manifest).read_text().splitlines() if line.strip())]
    try:
        check_runtime(cfg,tasks)
        print('SIMULATOR_PREFLIGHT_PASS',tasks,flush=True)
        import vpp2.libero.eval_libero_task_manifest as evaluator
        original_instantiate=evaluator.instantiate
        original_load=evaluator._load_model_checkpoint
        original_task=evaluator.run_single_task
        def instantiate(config,*a,**kw):
            begin=time.monotonic();result=original_instantiate(config,*a,**kw)
            if str(config.get('_target_','')).endswith('create_action_model'):
                report['model_construction_seconds']=time.monotonic()-begin
            return result
        def load(*a,**kw):
            begin=time.monotonic();result=original_load(*a,**kw)
            report['checkpoint_load_seconds']=time.monotonic()-begin
            return result
        def task(*a,**kw):
            if not report['tasks']:
                torch.cuda.synchronize()
                report['load_and_setup_seconds']=time.monotonic()-started
                report['setup_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                report['setup_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
                report['resident_allocated_gib']=torch.cuda.memory_allocated()/2**30
                report['gpu_name']=torch.cuda.get_device_name()
                report['gpu_total_gib']=torch.cuda.get_device_properties(0).total_memory/2**30
                save();(root/'ready').write_text(str(time.time()))
                deadline=time.monotonic()+1800
                while not (root.parent/'go').exists():
                    if time.monotonic()>deadline:raise TimeoutError('Phase barrier timed out')
                    time.sleep(.2)
                torch.cuda.reset_peak_memory_stats()
                report['rollout_started_at']=time.time()
                print('ROLLOUT_START',report['gpu_id'],flush=True)
            begin=time.monotonic()
            result=original_task(*a,**kw)
            torch.cuda.synchronize()
            report['tasks'].append(dict(suite=str(cfg.EVALUATION.task_suite_name),task_id=int(cfg.EVALUATION.task_id),episodes=int(cfg.EVALUATION.num_trials),seconds=time.monotonic()-begin,successes=int(result['successes'])))
            report['rollout_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            report['rollout_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            save()
            return result
        evaluator.instantiate=instantiate
        evaluator._load_model_checkpoint=load
        evaluator.run_single_task=task
        evaluator.eval_task_manifest(cfg)
        report['status']='PASS'
        report['rollout_finished_at']=time.time()
    except BaseException as e:
        report['status']='FAIL';report['error']=repr(e)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        if torch.cuda.is_initialized():
            report['final_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            report['final_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
        save()
        print('WORKER_PROFILE',json.dumps(report),flush=True)


if __name__=='__main__':main()
