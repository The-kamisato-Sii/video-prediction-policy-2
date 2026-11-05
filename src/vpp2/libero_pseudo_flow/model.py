"""Pinned 4RC encoder and official heads; decode only requested source/target pairs."""
import numpy as np
import torch
from PIL import Image


class FourRC:
    def __init__(self, checkpoint, device='cuda', resolution=504, pair_batch=4):
        from arc.models.arc.arc import Arc
        from safetensors.torch import load_file
        self.device=torch.device(device);self.resolution=resolution;self.pair_batch=pair_batch
        assert resolution%14==0
        self.model=Arc()
        state=load_file(str(checkpoint),device='cpu')
        self.model.load_state_dict(state,strict=True);del state
        self.model=self.model.eval().to(self.device)
        self.model.requires_grad_(False)

    def images(self,rgb,flip_x=True):
        arrays=[]
        for frame in rgb:
            if flip_x:frame=frame[:,::-1]
            image=Image.fromarray(frame).resize((self.resolution,self.resolution),Image.Resampling.BICUBIC if frame.shape[0]<=self.resolution else Image.Resampling.LANCZOS)
            arrays.append(np.asarray(image))
        return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2)[None].to(self.device,dtype=torch.float32)/255

    @torch.inference_mode()
    def encode(self,rgb,flip_x=True):
        from arc.models.arc.utils.transform import pose_encoding_to_extri_intri
        from arc.models.arc.utils.geometry import unproject_depth_map_to_point_map
        x=self.images(rgb,flip_x)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feats,_=self.model.backbone(x,ref_view_strategy='first')
        with torch.autocast('cuda',enabled=False):
            fpfeats=[tuple(t.float() for t in f) for f in feats]
            geometry=self.model.head(fpfeats,self.resolution,self.resolution,patch_start_idx=0)
            pose=self.model.cam_dec(fpfeats[-1][1])
            c2w,K=pose_encoding_to_extri_intri(pose,(self.resolution,self.resolution))
        c2w=c2w[0].float().cpu().numpy();K=K[0].float().cpu().numpy()
        if c2w.shape[-2]==3:
            c2w=np.concatenate([c2w,np.tile([[[0,0,0,1]]],(len(c2w),1,1))],axis=1)
        depth=geometry['depth'][0].float().cpu().numpy()
        points,_=unproject_depth_map_to_point_map(depth[...,None],np.linalg.inv(c2w)[:,:3],K)
        conf=geometry['depth_conf'][0].float().cpu().numpy()
        # Official decoder consumes the GLOBAL half of the four encoder stages.
        tokens=[torch.cat([f[1].unsqueeze(2),f[2].unsqueeze(2),f[0]],dim=2)[...,1536:] for f in feats]
        del fpfeats,feats,geometry
        return dict(x=x,tokens=tokens,points=points,confidence=conf,c2w=c2w,K=K)

    @torch.inference_mode()
    def pairs(self,encoded,pairs):
        """Same decoder, independent per target: pack (source,target) along batch.

        Source is always index 0 in each pair; return only target index 1.
        This does NOT subtract point maps at the same image pixel.
        """
        x=encoded['x'];outputs=[];confs=[]
        for offset in range(0,len(pairs),self.pair_batch):
            group=pairs[offset:offset+self.pair_batch]
            idx=torch.tensor(group,device=self.device,dtype=torch.long)
            xx=x[0,idx]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=[self.model.motion_decoder(t[0,idx],images=xx,patch_start_idx=2,track_query_idx=0)[:,1:2] for t in encoded['tokens']]
            with torch.autocast('cuda',enabled=False):
                flow,conf=self.model.track_head([h.float() for h in hidden],images=xx[:,1:2],patch_start_idx=1,frames_chunk_size=1)
            outputs.extend(flow[:,0].float().cpu().numpy());confs.extend(conf[:,0].float().cpu().numpy())
        return np.stack(outputs),np.stack(confs)

    @torch.inference_mode()
    def check_pair_parity(self,encoded,source=0,target=1):
        x=encoded['x'];n=x.shape[1]
        pairs=[(0,1),(0,min(4,n-1)),(min(3,n-2),min(4,n-1)),(min(3,n-2),min(7,n-1))]
        references={}
        for q in sorted({q for q,t in pairs}):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h=[self.model.motion_decoder(t,images=x,patch_start_idx=2,track_query_idx=q) for t in encoded['tokens']]
            with torch.autocast('cuda',enabled=False):
                original,_=self.model.track_head([t.float() for t in h],images=x,patch_start_idx=1,frames_chunk_size=1)
            for source_id,target_id in pairs:
                if source_id==q:references[(source_id,target_id)]=original[0,target_id].float().cpu().numpy()
            del h,original
        fast,_=self.pairs(encoded,pairs)
        ref=np.stack([references[pair] for pair in pairs])
        error=float(np.max(np.abs(fast-ref)))
        np.testing.assert_allclose(fast,ref,atol=3e-3,rtol=3e-2)
        return dict(passed=True,max_abs=error,mean_abs=float(np.mean(np.abs(fast-ref))),pairs=pairs,reference='Official full-target motion decoder on identical encoder features',atol=.003,rtol=.03)
