"""CPU contracts for axis conversion, source-pixel transport, scaling and LZ4."""
import tempfile
from pathlib import Path
import numpy as np
from vpp2.libero_pseudo_flow.geometry import resample,project,normalize_clip,write_record,read_record,atomic_json,load_clip

def main():
    native=256;size=504
    K=np.array([[420.,0,252.],[0,420.,252.],[0,0,1.]])
    y,x=np.mgrid[:size,:size]
    p=np.stack([(x-252)/420,(y-252)/420,np.ones_like(x)],axis=-1).astype('f4')
    output=resample(p,native)
    uv,_=project(output,np.eye(4),K,size,native)
    pixels=np.floor((np.arange(224)+.5)*native/224)
    expected=(pixels+.5)*224/native-.5
    ex,ey=np.meshgrid(expected,expected)
    assert np.max(np.abs(uv-np.stack([ex,ey],-1)))<.02
    translated=output+np.array([.1,0,0],dtype='f4')
    target,_=project(translated,np.eye(4),K,size,native)
    # Physical +X goes left in the stored mirrored RGB.
    assert np.all(target[...,0]<uv[...,0])
    points=np.stack([output+i*.001 for i in range(8)])
    flow=np.full_like(points,.03);valid=np.ones(points.shape[:-1],dtype=bool)
    normalized=normalize_clip(points,flow,valid)
    scaled=normalize_clip(points*3+np.array([2.,3.,4.]),flow*3,valid)
    np.testing.assert_allclose(normalized[0],scaled[0],atol=1e-5)
    np.testing.assert_allclose(normalized[1],scaled[1],atol=1e-5)
    assert abs(np.linalg.norm(normalized[0][valid],axis=-1).mean()-1)<1e-5
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'test.npz.lz4';arrays=dict(point_map=points,scene_flow=flow,valid=valid)
        write_record(p,arrays);loaded=read_record(p)
        for k in arrays:np.testing.assert_array_equal(arrays[k],loaded[k])
        episode=Path(tmp)/'episode';episode.mkdir()
        atomic_json(episode/'meta.json',dict(complete_episode=True,frames=33))
        for j,i in enumerate(range(0,32,4)):
            write_record(episode/f'frame_{i:06d}.npz.lz4',dict(point_map=points[j],scene_flow_dt1=flow[j],scene_flow_dt4=flow[j]*4,valid_mask=valid[j],flow_valid_mask_dt1=valid[j],flow_valid_mask_dt4=valid[j]))
        a=load_clip(episode,0,1);b=load_clip(episode,0,4)
        assert a['point_map'].shape==b['point_map'].shape==(8,224,224,3)
        np.testing.assert_array_equal(a['frame_index'],np.arange(8)*4)
        np.testing.assert_allclose(b['scene_flow'],a['scene_flow']*4,atol=1e-6)
        try:load_clip(episode,1)
        except IndexError:pass
        else:raise AssertionError('Tail exceeding action horizon was accepted')
    report=dict(passed=True,projection_max_pixel_error=float(np.max(np.abs(uv-np.stack([ex,ey],-1)))),normalization_similarity_invariant=True,lz4_lossless=True,physical_positive_x_projects_left_in_mirrored_rgb=True)
    atomic_json('runs/libero_pseudo_flow_4rc_20261102/geometry-tests.json',report)
    print(report)
if __name__=='__main__':main()
