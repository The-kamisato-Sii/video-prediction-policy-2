"""Compare identical PRO episodes serially and across two independent GPU policies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from omegaconf import OmegaConf
from vpp2.libero.evaluate import runtime_environment,worker_config,validate_checkpoint,check_idle_gpus

REPO=Path(__file__).resolve().parents[2]
CASES=[(suite+'_'+axis,0) for suite in ['libero_spatial','libero_object','libero_goal','libero_10'] for axis in ['swap','task']]


def run_phase(cfg,root,gpus):
    root.mkdir()
    processes=[]
    env_reports=[]
    try:
        for index,gpu in enumerate(gpus):
            profile=root/f'gpu{gpu}';profile.mkdir()
            manifest=profile/'tasks.txt';assigned=CASES[index::len(gpus)]
            manifest.write_text(''.join(f'{s},{t}\n' for s,t in assigned))
            c=worker_config(cfg,profile/'results',manifest,gpu,None)
            c.EVALUATION.num_trials=1
            c.EVALUATION.diagnostic_only=True
            c.release_benchmark='pro';c.release_benchmark_root=str(Path(cfg.benchmark_root).resolve())
            config=profile/'worker.yaml';OmegaConf.save(c,config)
            env=runtime_environment(cfg,gpu)
            # Independent policies, even if the scheduling container has torchrun variables.
            for key in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:
                env.pop(key,None)
            for key in list(env):
                if key.startswith('ACCELERATE_'):
                    env.pop(key)
            env.update(OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',VPP2_TORCH_LOAD_MMAP='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
            log=(profile/'worker.log').open('w')
            command=[sys.executable,str(REPO/'scripts/libero_debug/profile_pro_worker.py'),'--config',str(config),'--profile-dir',str(profile)]
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log,profile))
            env_reports.append(dict(gpu=gpu,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],egl_device=env['MUJOCO_EGL_DEVICE_ID'],tasks=assigned))
        (root/'allocation.json').write_text(json.dumps(env_reports,indent=2))
        deadline=time.monotonic()+1800
        while not all((p/'ready').exists() for _,_,p in processes):
            for process,_,profile in processes:
                if process.poll() is not None:raise RuntimeError(f'Worker {profile} exited {process.returncode} before ready')
            if time.monotonic()>deadline:raise TimeoutError('Model load/setup exceeded 30 minutes')
            time.sleep(2)
        start=time.time();(root/'go').write_text(str(start))
        print('PHASE_READY',root.name,'gpus',gpus,flush=True)
        while any(process.poll() is None for process,_,_ in processes):
            for process,_,profile in processes:
                if process.poll() not in [None,0]:raise RuntimeError(f'Worker {profile} exited {process.returncode}')
            time.sleep(2)
        reports=[]
        for process,_,profile in processes:
            if process.returncode!=0:raise RuntimeError(f'{profile} failed')
            report=json.loads((profile/'profile.json').read_text())
            assert report['status']=='PASS'
            reports.append(report)
        wall=max(x['rollout_finished_at'] for x in reports)-start
        result=dict(status='PASS',gpus=gpus,episodes=sum(t['episodes'] for r in reports for t in r['tasks']),rollout_wall_seconds=wall,episodes_per_hour=8/wall*3600,setup_wall_seconds=max(r['load_and_setup_seconds'] for r in reports),workers=reports)
        assert result['episodes']==len(CASES)
        (root/'summary.json').write_text(json.dumps(result,indent=2))
        print('PHASE_PASS',root.name,wall,'seconds',flush=True)
        return result
    finally:
        for process,log,_ in processes:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    p.add_argument('--phase',choices=['all','single_gpu','two_gpu'],default='all')
    args=p.parse_args()
    os.chdir(REPO)
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.load(REPO/'configs/libero/eval_pro.yaml')
    run=REPO/'runs/libero/libero-action-16gpu-b16-job-e252bb1f-b0fd-499c-bf63-fcc4f26d16c8-round0'
    cfg.action_checkpoint=str(run/'checkpoints/weights/step_030000.pt');cfg.training_config=str(run/'config.yaml')
    cfg.runtime_config_dir=str(root/'runtime');cfg.benchmark_root=str((REPO/cfg.benchmark_root).resolve())
    check_idle_gpus([0] if args.phase=='single_gpu' else [0,1]);validate_checkpoint(cfg)
    (root/'contract.json').write_text(json.dumps(dict(diagnostic_only=True,action_step=30000,video_step=10000,action_nfe=10,trials_per_case=1,init_state_index=1,cases=CASES,checkpoint=cfg.action_checkpoint,protocol_reference='public800'),indent=2))
    monitor_log=(root/'nvidia-smi.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu','--format=csv,noheader','--loop=2'],stdout=monitor_log,stderr=subprocess.STDOUT)
    try:
        if args.phase!='all':
            result=run_phase(cfg,root/args.phase,[0,1] if args.phase=='two_gpu' else [0])
            (root/'summary.json').write_text(json.dumps(result,indent=2))
            print('BENCHMARK_PHASE_PASS',args.phase,result['rollout_wall_seconds'],flush=True)
            return
        serial=run_phase(cfg,root/'single_gpu',[0])
        parallel=run_phase(cfg,root/'two_gpu',[0,1])
        # One task per suite/axis is a small timing sample, not a success-rate estimate.
        report=dict(status='PASS',single_gpu=serial,two_gpu=parallel,speedup=serial['rollout_wall_seconds']/parallel['rollout_wall_seconds'],estimated_public800_hours=parallel['rollout_wall_seconds']*100/3600,timing_sample_episodes=8,estimate_limitation='Only task0 of each of eight suite/axis cells, one initial state; episode lengths and success rates vary.')
        (root/'summary.json').write_text(json.dumps(report,indent=2));print('BENCHMARK_PASS',json.dumps({k:v for k,v in report.items() if k not in ['single_gpu','two_gpu']}),flush=True)
    finally:
        monitor.terminate();monitor.wait();monitor_log.close()


if __name__=='__main__':main()
