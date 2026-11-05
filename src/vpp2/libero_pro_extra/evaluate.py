"""Additional PRO axes, using the original rollout code and isolated assets."""
import argparse, fcntl, json, os, subprocess, sys, time, zipfile
from pathlib import Path
from omegaconf import OmegaConf
from vpp2.libero import evaluate as harness
from vpp2.libero.protocols import BASE_SUITES, PROTOCOLS, validate_config as baseline_validate
from vpp2.preflight import write_json_atomic

AXES=('object','lan','env')

def reference_config(cfg):
    c=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True));c.protocol='public800';return c

def validate_config(cfg):
    assert cfg.protocol=='extra1200_v1' and cfg.extra_axis in AXES and cfg.benchmark=='pro'
    p=dict(baseline_validate(reference_config(cfg)))
    p.update(name='extra1200_v1/'+cfg.extra_axis,episodes=400)
    return p

def cells(cfg):
    return [(None,s+'_'+cfg.extra_axis,t) for s in BASE_SUITES for t in range(10)]

def validate_checkpoint(cfg):
    validate_config(cfg)
    c=reference_config(cfg)
    if cfg.extra_model_kind=='flow':
        from vpp2.libero_flow.evaluate import validate_checkpoint as check
    else:check=harness.validate_checkpoint
    report=check(c)
    report.update(extra_protocol='extra1200_v1',extra_axis=cfg.extra_axis,
                  checkpoint_validation_uses_original_public800_rules=True)
    write_json_atomic(report,Path(cfg.output_dir).parent/'checkpoint-check.json')
    return report

def worker_config(cfg,manifest,gpu):
    c=harness.worker_config(reference_config(cfg),Path(cfg.output_dir)/'results/seedNone',manifest,gpu,None)
    if cfg.extra_model_kind=='flow':
        c.model._target_='vpp2.libero_flow.inference.create_flow_inference_model'
        c.model.flow_config=OmegaConf.to_container(c.flow,resolve=True)
        c.model.inference_audit_path=str(Path(cfg.output_dir).parent/f'inference-parity-gpu{gpu}.json')
    c.release_benchmark='pro';c.release_benchmark_root=str(Path(cfg.benchmark_root).resolve())
    return c

def audit(cfg):
    # Harness auditing is reused unchanged; only the expected cells/protocol differ.
    old_cells,old_validate=harness.cells,harness.validate_config
    try:
        harness.cells=lambda benchmark: cells(cfg)
        harness.validate_config=validate_config
        report=harness.audit(cfg)
    finally:harness.cells,harness.validate_config=old_cells,old_validate
    report['extra_axis']=cfg.extra_axis
    if cfg.extra_axis=='env':
        definition=json.loads(Path(cfg.extra_assets_definition).read_text())
        names={x['task'] for x in definition['environment_tasks'] if not x['effective_shift']}
        checks=json.loads(Path(cfg.extra_asset_check).read_text())
        controls={(x['suite'],x['task_id']) for x in checks['results'] if Path(x['bddl']).stem in names}
        effective=dict(episodes=0,successes=0);unchanged=dict(episodes=0,successes=0)
        for path in (Path(cfg.output_dir)/'results').rglob('*_results.json'):
            r=json.loads(path.read_text())
            dest=unchanged if (r['task_suite'],r['task_id']) in controls else effective
            dest['episodes']+=len(r['success_episodes'])+len(r['failure_episodes'])
            dest['successes']+=len(r['success_episodes'])
        for item in (effective,unchanged):item['success_rate']=item['successes']/item['episodes'] if item['episodes'] else None
        report.update(effective_environment_shift=effective,unchanged_environment_controls=unchanged)
        if report['complete']:assert effective['episodes']==350 and unchanged['episodes']==50
    return report

def run(cfg,gpu,resume=False):
    validate_config(cfg)
    gate=json.loads(Path(cfg.extra_asset_check).read_text())
    assert gate['status']=='PASS' and gate['checked_initial_states']==400 and gate['axis']==cfg.extra_axis
    root=Path(cfg.output_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A newly completed training job writes its final checkpoint directly.
        # Do not open it while the producer is still writing its ZIP archive.
        checkpoint=Path(cfg.action_checkpoint);deadline=time.monotonic()+600
        while True:
            st=checkpoint.stat()
            if time.time()-st.st_mtime>=60 and zipfile.is_zipfile(checkpoint):break
            if time.monotonic()>deadline:raise TimeoutError(f'Checkpoint not fully written: {checkpoint}')
            time.sleep(5)
        validate_checkpoint(cfg);harness.check_idle_gpus([gpu])
        name=f'seedNone_gpu{gpu}'
        config=OmegaConf.to_container(cfg,resolve=True)
        contract=dict(config=config,gpus=[gpu],jobs=[name],artifacts={
            k:dict(path=str(Path(config[k]).resolve()),size_bytes=Path(config[k]).stat().st_size,mtime_ns=Path(config[k]).stat().st_mtime_ns)
            for k in ('action_checkpoint','video_checkpoint','dataset_stats','training_config','extra_assets_definition','extra_asset_check')})
        previous=root/'RUN_CONFIG.json'
        if previous.exists():
            assert resume and json.loads(previous.read_text())==contract,'Resume contract changed'
        else:
            assert not resume;write_json_atomic(contract,previous)
        for d in ('manifests','logs','status','results'):(root/d).mkdir(exist_ok=True)
        manifest=root/'manifests'/f'{name}.txt'
        manifest.write_text(''.join(f'{s},{t}\n' for _,s,t in cells(cfg)))
        worker=worker_config(cfg,manifest,gpu);worker_path=root/'manifests'/f'{name}.yaml';OmegaConf.save(worker,worker_path)
        env=harness.runtime_environment(cfg,gpu)
        (root/'full_run.status').write_text('RUNNING\n');status=root/'status'/f'{name}.status';status.write_text('RUNNING\n')
        with (root/'logs'/f'{name}.log').open('a') as log:
            code=subprocess.run([sys.executable,'-u','-m','vpp2.libero_pro_extra.worker','--config',str(worker_path)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        status.write_text(str(code)+'\n')
        report=audit(cfg);write_json_atomic(report,root/'summary.json')
        success=code==0 and report['complete']
        (root/'full_run.status').write_text('0\n' if success else '1\n')
        assert success,f'Evaluation incomplete: {root}/summary.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    cfg=OmegaConf.load(a.config)
    if a.check_only:validate_checkpoint(cfg)
    else:run(cfg,a.gpu,a.resume)
if __name__=='__main__':main()
