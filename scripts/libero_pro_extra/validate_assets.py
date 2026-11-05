"""Generate frozen environment states, then validate all ten scored initial states."""
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
RUNTIME=OUT/'runtime'
# Use only the isolated runtime; never modify the original evaluation assets.
os.environ['LIBERO_CONFIG_PATH']=str(OUT/'runtime_config')
os.environ['LIBERO_TYPE']='pro';os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
sys.path[:0]=[str(RUNTIME),str(ROOT/'src')]
from omegaconf import OmegaConf
core=RUNTIME/'liberopro/liberopro'
c=OUT/'runtime_config';c.mkdir(exist_ok=True)
if not (c/'config.yaml').exists():
    OmegaConf.save(OmegaConf.create(dict(benchmark_root=str(core),bddl_files=str(core/'bddl_files'),init_states=str(core/'init_files'),assets=str(core/'assets'),datasets=str(core.parent/'datasets'))),c/'config.yaml')
from vpp2.libero_pro_extra.runtime import register_environment_suites
register_environment_suites()
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vpp2.libero.eval_helpers import load_trusted_task_init_states
import torch,numpy as np,random

def run(axis, only_base=None):
    dest=OUT/'asset_checks';dest.mkdir(exist_ok=True)
    result=[]
    for base in ('libero_spatial','libero_object','libero_goal','libero_10'):
        if only_base and base!=only_base: continue
        suite=benchmark.get_benchmark_dict()[base+'_'+axis]()
        for tid in range(10):
            task=suite.get_task(tid); t0=time.time(); itemfile=dest/f'{base}_{axis}_{tid}.json'
            if itemfile.exists() and json.loads(itemfile.read_text()).get('status')=='PASS':
                result.append(json.loads(itemfile.read_text()));continue
            path=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            initial=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
            env=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=256,camera_widths=256,hard_reset=(axis!='env'))
            try:
                if axis=='env' and not initial.exists():
                    states=[]
                    for seed in range(50):
                        random.seed(seed);np.random.seed(seed);env.seed(seed);env.reset()
                        state=np.array(env.get_sim_state(),copy=True)
                        assert np.isfinite(state).all()
                        states.append(state)
                    tmp=initial.with_suffix('.tmp');torch.save(np.array(states),tmp);tmp.replace(initial)
                states=load_trusted_task_init_states(suite,tid);assert len(states)==50
                checks=[]
                for index in range(1,11):
                    env.seed(index);env.reset();env.set_init_state(states[index])
                    for _ in range(15):obs,_,_,_=env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
                    assert np.isfinite(env.get_sim_state()).all()
                    for k in ('agentview_image','robot0_eye_in_hand_image'):
                        assert obs[k].shape==(256,256,3) and obs[k].std()>1
                    checks.append(dict(index=index,initial_success=bool(env.check_success())))
                item=dict(status='PASS',suite=base+'_'+axis,task_id=tid,language=task.language,
                          bddl=str(path),init_file=str(initial),states=50,checked=checks,seconds=time.time()-t0)
                itemfile.write_text(json.dumps(item,indent=2)+'\n'); result.append(item)
                print('PASS',base,axis,tid,round(time.time()-t0,1),flush=True)
            finally:env.close()
    if only_base:
        assert len(result)==10
        (dest/f'{axis}_{only_base}.json').write_text(json.dumps(dict(status='PASS',results=result),indent=2)+'\n')
        paths=[dest/f'{axis}_{b}.json' for b in ('libero_spatial','libero_object','libero_goal','libero_10')]
        if not all(p.exists() for p in paths): return
        result=[r for p in paths for r in json.loads(p.read_text())['results']]
    assert len(result)==40
    report=dict(status='PASS',axis=axis,tasks=40,checked_initial_states=400,
                initial_successes=sum(x['initial_success'] for r in result for x in r['checked']),results=result)
    (dest/f'{axis}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('AXIS_PASS',axis,'initial_successes',report['initial_successes'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--axis',required=True,choices=['object','lan','env']);p.add_argument('--base');a=p.parse_args();run(a.axis,a.base)
