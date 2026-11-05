"""Project estimated material motion into RGB; GT is read for visualization only."""
import argparse,json
from pathlib import Path
import numpy as np
from .geometry import read_record,project,normalize_clip,atomic_json
from .extract import decode


def draw(rgb,uv,endpoint,valid,title):
    import cv2
    image=rgb.copy();h,w=image.shape[:2]
    yy,xx=np.mgrid[8:h:12,8:w:12]
    norm=np.linalg.norm(endpoint-uv,axis=-1)
    visible=valid[yy,xx]&np.isfinite(norm[yy,xx])&(norm[yy,xx]>.65)&(norm[yy,xx]<h/2)
    for y,x in zip(yy[visible],xx[visible]):
        a=tuple(np.rint(uv[y,x]).astype(int));b=tuple(np.rint(endpoint[y,x]).astype(int))
        hue=np.uint8([[[int(x/w*170),220,255]]]);color=tuple(int(v) for v in cv2.cvtColor(hue,cv2.COLOR_HSV2RGB)[0,0])
        cv2.arrowedLine(image,a,b,color,1,cv2.LINE_AA,tipLength=.25)
    cv2.rectangle(image,(0,0),(w,22),(20,20,20),-1)
    cv2.putText(image,title,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
    return image


def main():
    import av,cv2
    p=argparse.ArgumentParser();p.add_argument('--episode-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--dt',type=int,default=4);a=p.parse_args()
    d=Path(a.episode_dir);meta=json.loads((d/'meta.json').read_text());rgb=decode(meta['source_video']);native=meta['native_resolution'];size=meta['model_resolution'];n=meta['frames']
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    # Inspect data validity and exact normalization separately from display filtering.
    records=[read_record(d/f'frame_{i:06d}.npz.lz4') for i in range(n)]
    ids=np.arange(8)*4
    point=np.stack([records[i]['point_map'] for i in ids]);flow=np.stack([records[i][f'scene_flow_dt{a.dt}'] for i in ids]);valid=np.stack([records[i]['valid_mask'] for i in ids]);fv=np.stack([records[i][f'flow_valid_mask_dt{a.dt}'] for i in ids])
    pn,fn,mu,scale=normalize_clip(point,flow,valid,fv)
    stats=dict(mu=mu.tolist(),S=scale,normalized_point_mean=pn[valid].mean(0).tolist(),normalized_mean_radius=float(np.linalg.norm(pn[valid],axis=-1).mean()),point_quantiles=np.quantile(pn[valid],[0,.01,.5,.99,1]).tolist(),flow_norm_quantiles=np.quantile(np.linalg.norm(fn[fv],axis=-1),[0,.5,.9,.99,1]).tolist(),valid_fraction=float(valid.mean()),flow_valid_fraction=float(fv.mean()),codec_flow_multiplier_applied=False)
    assert abs(stats['normalized_mean_radius']-1)<1e-5
    assert np.max(np.abs(stats['normalized_point_mean']))<1e-4
    atomic_json(target.with_suffix('.json'),stats)
    container=av.open(str(target),'w');stream=container.add_stream('libx264',rate=20);stream.width=224*2;stream.height=224;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
    contact=[];projection_errors=[]
    for i in range(n-a.dt):
        r=records[i];pmap=r['point_map'];vectors=r[f'scene_flow_dt{a.dt}']
        uv,z=project(pmap,r['source_c2ref'],r['source_intrinsic'],size,native)
        end,z2=project(pmap+vectors,r[f'target_c2ref_dt{a.dt}'],r[f'target_intrinsic_dt{a.dt}'],size,native)
        image=cv2.resize(rgb[i],(224,224),interpolation=cv2.INTER_AREA)
        left=draw(image,uv,uv,np.zeros((224,224),bool),f'RGB frame {i}')
        right=draw(image,uv,end,r['valid_mask']&r[f'flow_valid_mask_dt{a.dt}']&z&z2,f'4RC: t -> t+{a.dt} (1x arrows)')
        joined=np.concatenate([left,right],axis=1)
        if i in (0,4,8,12,16,20,24,28):contact.append(joined)
        for packet in stream.encode(av.VideoFrame.from_ndarray(joined,format='rgb24')):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close()
    sheet=np.concatenate(contact,axis=0);cv2.imwrite(str(target.with_suffix('.jpg')),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    print(json.dumps(dict(video=str(target),normalization=stats)),flush=True)

if __name__=='__main__':main()
