"""Validate full retained trajectories and both RGB streams without publishing labels."""
import argparse,json,time,traceback
from pathlib import Path
import h5py,numpy as np,mujoco
from PIL import Image
from vpp2.libero.motion_replay import replay,REPLAY_VERSION
from vpp2.libero.motion_extract import set_state,decode_rgb,orient,CAMERAS


def validate_rgb(dataset,episode,trajectory,output):
    with h5py.File(trajectory,'r') as f:
        model=mujoco.MjModel.from_xml_string(f.attrs['model_file'])
        states=f['states'][:];render_data={k:v[:] for k,v in f['render_data'].items()}
        proprio=f['proprio_errors'][:].max(axis=0).tolist()
        reset_count=int(f.attrs['reset_count'])
    model.vis.global_.offheight=512;model.vis.global_.offwidth=512
    data=mujoco.MjData(model);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1
    summary={}
    with mujoco.Renderer(model,512,512) as renderer:
        for key,camera in CAMERAS.items():
            rgb=decode_rgb(dataset/f'videos/chunk-{episode//1000:03d}/observation.images.{key}/episode_{episode:06d}.mp4')
            if len(rgb)!=len(states):raise ValueError('Video/state length mismatch')
            errors=[];worst=None;orientation=None
            for i,state in enumerate(states):
                set_state(model,data,state,render_data,i);renderer.update_scene(data,camera=camera,scene_option=opt)
                raw=renderer.render().copy()
                if orientation is None:
                    orientation=min(('identity','flip_x','flip_y','rotate180'),key=lambda name:np.abs(orient(raw,name).astype(float)-rgb[i]).mean())
                rendered=orient(raw,orientation);difference=np.abs(rendered.astype(float)-rgb[i]);mae=float(difference.mean())
                if not errors or mae>max(errors):worst=(i,rgb[i].copy(),rendered.copy(),difference)
                errors.append(mae)
            peak,original,rendered,difference=worst
            summary[key]=dict(mean=float(np.mean(errors)),max=max(errors),worst_frame=peak,frames_over_3=int(np.sum(np.array(errors)>3)),orientation=orientation)
            if max(errors)>3:
                Image.fromarray(np.concatenate([original,rendered,np.clip(difference*8,0,255).astype('uint8')],axis=1)).save(output/f'worst_{episode:06d}_{key}.png')
            np.save(output/f'rgb_errors_{episode:06d}_{key}.npy',errors)
    return dict(frames=len(states),max_proprio_error=proprio,reset_count=reset_count,rgb=summary,
                passed=all(x['max']<=3 for x in summary.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--version-dir',default='v3');a=p.parse_args()
    root=Path.cwd();run=root/'runs/motion_validation_20261023';out=run/a.version_dir/a.suite;out.mkdir(exist_ok=True,parents=True)
    rows=json.loads((run/'selection.json').read_text())[a.suite]
    results=[]
    for row in rows:
        ep=row['episode'];report=out/f'episode_{ep:06d}.json'
        if report.exists():results.append(json.loads(report.read_text()));continue
        started=time.monotonic();result=dict(row)
        print('START',a.suite,ep,row['demo'],'removed_noops',row['removed'],flush=True)
        try:
            ds=root/f'data/libero/libero_{a.suite}_no_noops_lerobot'
            source=root/f'data/libero_motion_sources/libero_{a.suite}_no_noops_lerobot_episode_{ep:06d}.hdf5'
            trajectory=out/f'replayed_episode_{ep:06d}.hdf5'
            if not trajectory.exists():replay(ds,ep,source,trajectory)
            result.update(validate_rgb(ds,ep,trajectory,out));result['replay_version']=REPLAY_VERSION
        except Exception as exc:
            result.update(passed=False,error=str(exc));traceback.print_exc()
        result['seconds']=time.monotonic()-started
        report.write_text(json.dumps(result,indent=2));results.append(result)
        (out/'summary.json').write_text(json.dumps(results,indent=2))
        print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':main()
