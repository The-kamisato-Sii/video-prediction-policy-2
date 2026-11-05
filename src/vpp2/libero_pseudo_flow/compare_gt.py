"""Read-only GT comparison after RGB-only estimation, never part of extraction."""
import argparse,json
from pathlib import Path
import numpy as np
from .geometry import read_record,project,normalize_clip,atomic_json
from .extract import decode
from .visualize import draw


def main():
    import h5py,av,cv2
    p=argparse.ArgumentParser();p.add_argument('--episode-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.episode_dir);m=json.loads((folder/'meta.json').read_text());video=Path(m['source_video']);dataset=video.parents[3]
    gt=dataset/f"motions/chunk-{m['episode']//1000:03d}/episode_{m['episode']:06d}.hdf5"
    if not gt.exists():print('No GT for this RGB episode; pseudo-only visualization remains available');return
    out=Path(a.output);rgb=decode(video);records=[read_record(folder/f'frame_{i:06d}.npz.lz4') for i in range(m['frames'])]
    ids=np.arange(8)*4
    with h5py.File(gt,'r') as f:
        g=f['image'];P=g['point_map'][ids];V=g['flow_dt4/scene_flow'][ids];mask=g['valid_mask'][ids]&g['flow_dt4/valid_mask'][ids]
        pp=np.stack([records[i]['point_map'] for i in ids]);vv=np.stack([records[i]['scene_flow_dt4'] for i in ids]);pm=np.stack([records[i]['valid_mask']&records[i]['flow_valid_mask_dt4'] for i in ids])
        pn,fn,mu,s=normalize_clip(pp,vv,pm)
        pg,fg,mg,sg=normalize_clip(P,V,mask)
        common=mask&pm;moving=common&(np.linalg.norm(fg,axis=-1)>.01);static=common&(np.linalg.norm(fg,axis=-1)<1e-5)
        epe=np.linalg.norm(fn-fg,axis=-1);cos=np.sum(fn*fg,axis=-1)/(np.linalg.norm(fn,axis=-1)*np.linalg.norm(fg,axis=-1)+1e-8)
        metrics=dict(gt_used_only_for_evaluation=True,normalization='Each label source uses its own head-view clip mean and radius; no fitted alignment',normalized_point_l1=float(np.abs(pn[common]-pg[common]).mean()),normalized_flow_epe_mean=float(epe[common].mean()),normalized_flow_epe_moving=float(epe[moving].mean()) if moving.any() else None,moving_cosine_mean=float(cos[moving].mean()) if moving.any() else None,static_predicted_motion_norm=float(np.linalg.norm(fn[static],axis=-1).mean()) if static.any() else None,gt_moving_pixels=int(moving.sum()),gt_static_pixels=int(static.sum()))
        atomic_json(out.with_suffix('.json'),metrics)
        container=av.open(str(out),'w');stream=container.add_stream('libx264',rate=20);stream.width=672;stream.height=224;stream.pix_fmt='yuv420p';stream.options={'crf':'18'};sheet=[]
        for i in range(m['frames']-4):
            r=records[i];image=cv2.resize(rgb[i],(224,224),interpolation=cv2.INTER_AREA)
            pu,pz=project(r['point_map'],r['source_c2ref'],r['source_intrinsic'],m['model_resolution'],m['native_resolution'])
            pe,pez=project(r['point_map']+r['scene_flow_dt4'],r['target_c2ref_dt4'],r['target_intrinsic_dt4'],m['model_resolution'],m['native_resolution'])
            gp=g['point_map'][i];gv=g['flow_dt4/scene_flow'][i];k=g['intrinsic_native'][:]
            gu,gz=project(gp,g['camera_to_reference'][i],k,m['native_resolution'],m['native_resolution'])
            ge,gez=project(gp+gv,g['camera_to_reference'][i+4],k,m['native_resolution'],m['native_resolution'])
            views=[draw(image,pu,pu,np.zeros((224,224),bool),f'RGB {i}'),draw(image,gu,ge,g['valid_mask'][i]&gz&gez,'Simulator GT: dt4, 1x'),draw(image,pu,pe,r['valid_mask']&pz&pez,'4RC estimated: dt4, 1x')]
            combined=np.concatenate(views,axis=1)
            if i in (0,4,8,12,16,20,24,28):sheet.append(combined)
            for packet in stream.encode(av.VideoFrame.from_ndarray(combined,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
        container.close();cv2.imwrite(str(out.with_suffix('.jpg')),cv2.cvtColor(np.concatenate(sheet,axis=0),cv2.COLOR_RGB2BGR))
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
