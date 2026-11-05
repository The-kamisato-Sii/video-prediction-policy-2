"""Coordinate and storage helpers, independent of simulator labels."""
import io,json
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def write_record(path, arrays):
    import lz4.frame
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO();np.savez(buf,**arrays)
    temp=path.with_suffix('.tmp')
    temp.write_bytes(lz4.frame.compress(buf.getvalue(),compression_level=0))
    temp.replace(path)


def read_record(path):
    import lz4.frame
    with np.load(io.BytesIO(lz4.frame.decompress(Path(path).read_bytes())),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def sample_grid(native, output, model):
    # Same native pixel centers selected by original simulator extraction.
    pixels=np.minimum(((np.arange(output)+.5)*native/output).astype(int),native-1)
    xy=(pixels+.5)*model/native-.5
    return pixels,xy


def resample(array, native, output=224, flip_x=True):
    import cv2
    h,w=array.shape[:2];assert h==w
    _,xy=sample_grid(native,output,h)
    xx,yy=np.meshgrid((h-1-xy) if flip_x else xy,xy)
    return cv2.remap(np.asarray(array,dtype=np.float32),xx.astype('f4'),yy.astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def project(points, c2ref, K, model_size, native, output=224, flip_x=True):
    # Camera pose and intrinsics are predicted, not simulator calibration.
    cam=(points-c2ref[:3,3]) @ c2ref[:3,:3]
    pix=cam @ K.T
    uv=pix[...,:2]/np.maximum(pix[...,2:],1e-8)
    uv=(uv+.5)*native/model_size-.5
    if flip_x:uv[...,0]=native-1-uv[...,0]
    # Drawing on a conventionally resized RGB frame.
    uv=(uv+.5)*output/native-.5
    return uv,cam[...,2]>0


def normalize_clip(points, flow, valid, flow_valid=None):
    from vpp2.libero.motion_geometry import canonical_normalization
    mu,scale=canonical_normalization(points,valid)
    p=(points-mu)/scale;v=flow/scale
    p[~valid]=0;v[~(valid if flow_valid is None else flow_valid)]=0
    return p.astype('f4'),v.astype('f4'),mu.astype('f4'),float(scale)


def load_clip(episode_dir, start, dt=4):
    """Eight source frames 0,4,...28; raw flow is never pre-multiplied by 10."""
    directory=Path(episode_dir)
    meta=json.loads((directory/'meta.json').read_text())
    if not meta.get('complete_episode'):raise ValueError('Incomplete pseudo episode')
    ids=start+np.arange(8)*4
    if start<0 or start+32>=meta['frames']:raise IndexError('Same full-horizon policy as GT')
    records=[read_record(directory/f'frame_{i:06d}.npz.lz4') for i in ids]
    points=np.stack([r['point_map'] for r in records]);flow=np.stack([r[f'scene_flow_dt{dt}'] for r in records])
    valid=np.stack([r['valid_mask'] for r in records]);fv=np.stack([r[f'flow_valid_mask_dt{dt}'] for r in records])
    p,f,mu,scale=normalize_clip(points,flow,valid,fv)
    return dict(point_map=p,scene_flow=f,valid_mask=valid,flow_valid_mask=fv,center=mu,scale=scale,frame_index=ids,target_frame_index=ids+dt)
