"""MotionCrafter-style RGB overlay of source-frame material point trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
from .geometry import read_record,project,atomic_json
from .extract import decode


def main():
    import cv2,av
    p=argparse.ArgumentParser();p.add_argument('--episode-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.episode_dir);m=json.loads((folder/'meta.json').read_text());rgb=decode(m['source_video']);frames=m['frames']
    yy,xx=np.mgrid[6:224:10,6:224:10];yy=yy.ravel();xx=xx.ravel()
    tracks=[];valid=[]
    for i in range(frames):
        r=read_record(folder/f'frame_{i:06d}.npz.lz4')
        assert 'anchor_endpoint' in r
        uv,z=project(r['anchor_endpoint'],r['source_c2ref'],r['source_intrinsic'],m['model_resolution'],m['native_resolution'])
        tracks.append(uv[yy,xx]);valid.append(z[yy,xx]&np.isfinite(uv[yy,xx]).all(-1))
    tracks=np.stack(tracks);valid=np.stack(valid)
    displacement=np.linalg.norm(tracks-tracks[:1],axis=-1).max(0)
    keep=valid.all(0)&(displacement>2)&(displacement<150)
    ids=np.flatnonzero(keep)
    # Fixed grid, estimated motion only. Display selection never changes labels.
    if len(ids)>90:ids=ids[np.argsort(displacement[ids])[-90:]]
    target=Path(a.output);c=av.open(str(target),'w');s=c.add_stream('libx264',rate=20);s.width=448;s.height=224;s.pix_fmt='yuv420p';s.options={'crf':'18'}
    contact=[]
    for t in range(frames):
        image=cv2.resize(rgb[t],(224,224),interpolation=cv2.INTER_AREA);overlay=image.copy()
        for k in ids:
            for j in range(max(1,t-12),t+1):
                aa=tuple(np.rint(tracks[j-1,k]).astype(int));bb=tuple(np.rint(tracks[j,k]).astype(int))
                hsv=np.uint8([[[int(170*j/max(frames-1,1)),235,255]]]);color=tuple(map(int,cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)[0,0]))
                cv2.line(overlay,aa,bb,color,1,cv2.LINE_AA)
            point=tuple(np.rint(tracks[t,k]).astype(int));cv2.circle(overlay,point,2,(255,255,255),-1,cv2.LINE_AA)
        for im,label in [(image,f'RGB | frame {t}'),(overlay,'4RC 3D tracks -> 2D')]:
            cv2.rectangle(im,(0,0),(224,22),(20,20,20),-1);cv2.putText(im,label,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        joined=np.concatenate([image,overlay],axis=1)
        if t in (0,4,8,12,16,20,24,28,32):contact.append(joined)
        for packet in s.encode(av.VideoFrame.from_ndarray(joined,format='rgb24')):c.mux(packet)
    for packet in s.encode():c.mux(packet)
    c.close();cv2.imwrite(str(target.with_suffix('.jpg')),cv2.cvtColor(np.concatenate(contact,axis=0),cv2.COLOR_RGB2BGR))
    atomic_json(target.with_suffix('.json'),dict(source_frame=0,frames=frames,displayed_tracks=len(ids),method='Direct 4RC query of the same source pixels at each target time; not integration of 2D flow',display_filter='Fixed pixel grid; >2px trajectory movement; at most90; filtering affects display only'))
    print(target,flush=True)
if __name__=='__main__':main()
