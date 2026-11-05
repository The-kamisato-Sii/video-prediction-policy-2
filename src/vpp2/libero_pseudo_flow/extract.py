"""RGB-only 4RC extraction. Outputs live exclusively under pseudo_flow/."""
import argparse,fcntl,json,os,socket,time,traceback,hashlib
from pathlib import Path
import numpy as np
from .geometry import atomic_json,write_record,resample

SUITES={'spatial':'libero_spatial_no_noops_lerobot','object':'libero_object_no_noops_lerobot','goal':'libero_goal_no_noops_lerobot','long':'libero_10_no_noops_lerobot'}


def decode(path):
    import av
    with av.open(str(path)) as c:return np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])


def extract_episode(model,dataset,episode,length,output,args,parity=False):
    import torch
    starttime=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    video=dataset/f'videos/chunk-{episode//1000:03d}/observation.images.image/episode_{episode:06d}.mp4'
    rgb=decode(video);assert len(rgb)==length,(len(rgb),length)
    original_length=len(rgb)
    if args.max_frames:rgb=rgb[:args.max_frames]
    n,native=len(rgb),rgb.shape[1];assert rgb.shape[1]==rgb.shape[2]
    anchor_reference=None;windows=[];norm_sum=np.zeros(3);valid_count=0;flow_norms=[]
    for start in range(0,n,args.block):
        end=min(start+args.block,n)
        ids=sorted(set([0,*range(start,min(end+4,n))]))
        local={f:i for i,f in enumerate(ids)}
        torch.cuda.reset_peak_memory_stats()
        tic=time.time();enc=model.encode(rgb[ids],flip_x=True)
        encode_seconds=time.time()-tic
        C=enc['c2w'];anchor=C[0];rotation=anchor[:3,:3]
        points=(enc['points']-anchor[:3,3]) @ rotation
        anchor_depth=points[0,...,2].copy()
        if anchor_reference is None:
            anchor_reference=anchor_depth.copy();scale=1.0
        else:
            mask=np.isfinite(anchor_depth)&np.isfinite(anchor_reference)&(anchor_depth>1e-6)&(anchor_reference>1e-6)
            if mask.mean()<.9:raise ValueError('Insufficient shared anchor geometry')
            scale=float(np.median(anchor_reference[mask]/anchor_depth[mask]))
            if not np.isfinite(scale) or not .05<scale<20:raise ValueError(f'Unstable predicted scale: {scale}')
        points*=scale
        c2ref=np.linalg.inv(anchor)[None] @ C
        c2ref[:,:3,3]*=scale
        anchor_disagreement=float(np.median(np.abs(anchor_depth*scale-anchor_reference))/np.median(anchor_reference))
        if parity and start==0:
            atomic_json(output/'decoder_parity.json',model.check_pair_parity(enc,0,min(4,len(ids)-1)))
        requests=[(f,dt) for f in range(start,end) for dt in (1,4) if f+dt<n]
        pairs=[(local[f],local[f+dt]) for f,dt in requests]
        pair_start=time.time()
        if pairs:flows,conf=model.pairs(enc,pairs)
        pair_seconds=time.time()-pair_start
        by_pair={key:i for i,key in enumerate(requests)}
        anchor_tracks=None
        if args.preview_tracks:
            af,ac=model.pairs(enc,[(0,local[f]) for f in range(start,end)])
            anchor_tracks=points[0][None]+(af @ rotation)*scale
            if start==0:anchor_tracks[0]=points[0]
        writing_start=time.time()
        for f in range(start,end):
            q=local[f]
            p=resample(points[q],native)
            geometry_conf=resample(enc['confidence'][q],native)
            source_depth=resample(((enc['points'][q]-C[q,:3,3]) @ C[q,:3,:3])[...,2],native)
            valid=np.isfinite(p).all(-1)&np.isfinite(geometry_conf)&(geometry_conf>0)&np.isfinite(source_depth)&(source_depth>0)
            if valid.mean()<.95:raise ValueError(f'Invalid geometry coverage: frame {f}: {valid.mean()}')
            p[~valid]=0
            record=dict(point_map=p.astype('f4'),valid_mask=valid,depth_confidence=np.minimum(geometry_conf,65504).astype('f2'),frame_index=np.array(f,dtype='i4'),source_c2ref=c2ref[q].astype('f4'),source_intrinsic=enc['K'][q].astype('f4'),window_index=np.array(len(windows),dtype='i4'))
            for dt in (1,4):
                key=(f,dt)
                if key in by_pair:
                    j=by_pair[key]
                    v=resample((flows[j] @ rotation)*scale,native)
                    fc=resample(conf[j],native)
                    fv=valid&np.isfinite(v).all(-1)&np.isfinite(fc)&(fc>0)
                    if fv.mean()<.95:raise ValueError(f'Invalid flow coverage: {f},dt{dt}')
                    v[~fv]=0
                    target_c=c2ref[local[f+dt]];target_k=enc['K'][local[f+dt]]
                    flow_norms.append(float(np.mean(np.linalg.norm(v[fv],axis=-1))))
                else:
                    v=np.zeros_like(p);fc=np.zeros(p.shape[:2],dtype='f4');fv=np.zeros(p.shape[:2],bool);target_c=c2ref[q];target_k=enc['K'][q]
                record.update({f'scene_flow_dt{dt}':v.astype('f4'),f'flow_valid_mask_dt{dt}':fv,f'motion_confidence_dt{dt}':np.minimum(fc,65504).astype('f2'),f'target_c2ref_dt{dt}':target_c.astype('f4'),f'target_intrinsic_dt{dt}':target_k.astype('f4')})
            if anchor_tracks is not None:
                record['anchor_endpoint']=resample(anchor_tracks[f-start],native).astype('f4')
                record['anchor_confidence']=resample(ac[f-start],native).astype('f4')
            write_record(output/f'frame_{f:06d}.npz.lz4',record)
            norm_sum+=p[valid].sum(0,dtype=np.float64);valid_count+=int(valid.sum())
        del enc,points
        w=dict(source_start=start,source_end=end,encoder_frame_indices=ids,scale_to_first_window=scale,anchor_relative_depth_disagreement=anchor_disagreement,seconds=time.time()-tic,encode_seconds=encode_seconds,pair_decode_seconds=pair_seconds,storage_seconds=time.time()-writing_start,peak_gpu_memory_GiB=torch.cuda.max_memory_allocated()/2**30)
        windows.append(w)
        atomic_json(output/'progress.json',dict(state='RUNNING',episode=episode,frames_written=end,frames=n,elapsed_seconds=time.time()-starttime,windows=windows))
        print(json.dumps(dict(episode=episode,frames=f'{end}/{n}',window_seconds=round(w['seconds'],2))),flush=True)
    report=dict(schema='4rc_material_flow_v1',complete_episode=not args.max_frames or n==original_length,episode=episode,frames=n,original_frames=original_length,resolution=224,native_resolution=native,model_resolution=args.resolution,camera='image',model='Luo-Yihang/4RC',checkpoint=str(Path(args.checkpoint).resolve()),model_revision=json.loads((Path(args.checkpoint).parent/'source.json').read_text()),code_revision=args.code_revision,coordinate_frame='estimated_initial_agentview_opencv',units='4rc_relative_scale_not_metres',image_orientation='flip_x',flow_definition='source-pixel material-point displacement P(t->t+dt)-P(t); no camera ego-motion term',flow_dt=[1,4],raw_storage='unscaled float32 point_map and displacement; no x10 or latent scale baked in',normalization='load_clip: predicted head-view valid points across 8 source frames, mean center and isotropic mean radius; P=(P-mu)/S,F=F/S',temporal_sampling='retained LeRobot indices; training source offsets 0,4,...28; displacement targets source+dt',window_alignment='Each window includes RGB frame0; transform into its predicted OpenCV axes; one median anchor-depth scale shared by points and vectors',windows=windows,uses_gt_labels=False,uses_actions=False,occlusion_policy='Target occlusion does not invalidate material-point displacement; confidence retained separately',sample_policy='Finite geometry/motion with positive model confidence; no heuristic static-object masking',mean_flow_norm=float(np.mean(flow_norms)),elapsed_seconds=time.time()-starttime,hostname=socket.gethostname(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),source_video=str(video),config_id=args.config_id)
    atomic_json(output/'meta.json',report)
    atomic_json(output/'progress.json',dict(state='COMPLETE' if report['complete_episode'] else 'PREVIEW',frames_written=n,frames=n,elapsed_seconds=report['elapsed_seconds']))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--resolution',type=int,default=504);p.add_argument('--block',type=int,default=16);p.add_argument('--pair-batch',type=int,default=4);p.add_argument('--episode',type=int);p.add_argument('--max-frames',type=int);p.add_argument('--preview-root');p.add_argument('--status',required=True);p.add_argument('--code-revision',required=True);p.add_argument('--parity',action='store_true');p.add_argument('--preview-tracks',action='store_true');a=p.parse_args()
    dataset=Path('data/libero')/SUITES[a.suite]
    cfg=dict(model=a.code_revision,checkpoint=str(Path(a.checkpoint).resolve()),resolution=a.resolution,block=a.block,flow_dt=[1,4],flip_x=True,schema='4rc_material_flow_v1')
    a.config_id=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
    rows=[json.loads(l) for l in (dataset/'meta/episodes.jsonl').read_text().splitlines() if l.strip()]
    if a.episode is not None:rows=[r for r in rows if r['episode_index']==a.episode]
    if a.max_frames and not a.preview_root:raise ValueError('Partial previews must be outside production pseudo_flow')
    from .model import FourRC
    import torch
    torch.set_num_threads(4)
    atomic_json(a.status,dict(state='LOADING_MODEL',suite=a.suite,hostname=socket.gethostname(),config=cfg,started=time.time()))
    model=FourRC(a.checkpoint,resolution=a.resolution,pair_batch=a.pair_batch)
    done=[];failed=[];skipped=0
    for row in sorted(rows,key=lambda r:(-r['length'],r['episode_index'])):
        eid=int(row['episode_index'])
        parent=Path(a.preview_root) if a.preview_root else dataset/'pseudo_flow/4rc_v1'
        dest=parent/f'episode_{eid:06d}';dest.mkdir(parents=True,exist_ok=True)
        lock=(dest/'generation.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        try:
            if (dest/'meta.json').exists():
                meta=json.loads((dest/'meta.json').read_text())
                if meta.get('complete_episode'):
                    assert meta['config_id']==a.config_id,'Existing pseudo labels use different settings'
                    skipped+=1;continue
            atomic_json(a.status,dict(state='RUNNING',suite=a.suite,episode=eid,completed=done,failed=failed,skipped=skipped,config=cfg,updated=time.time()))
            report=extract_episode(model,dataset,eid,int(row['length']),dest,a,parity=a.parity and not done)
            done.append(dict(episode=eid,seconds=report['elapsed_seconds'],frames=report['frames']))
        except Exception as e:
            traceback.print_exc();failed.append(dict(episode=eid,error=repr(e)));atomic_json(dest/'error.json',failed[-1])
            if not done:raise
        finally:lock.close()
    atomic_json(a.status,dict(state='COMPLETE' if not failed else 'COMPLETE_WITH_ERRORS',suite=a.suite,completed=done,failed=failed,skipped=skipped,config=cfg,updated=time.time()))

if __name__=='__main__':main()
