"""Checkpoint admission must permit equivalent training GPU allocations."""
import pytest
import torch
from omegaconf import OmegaConf
from vpp2.libero.evaluate import validate_checkpoint

@pytest.mark.parametrize('batch,reference,accepted', [(16,256,True),(32,256,True),(8,256,True),(16,128,False),(3,256,False)])
def test_checkpoint_global_batch_contract(monkeypatch,batch,reference,accepted):
    cfg=OmegaConf.load('configs/libero/train_action.yaml')
    cfg.batch_size=batch
    cfg.reference_global_batch_size=reference
    cfg.gradient_accumulation_steps=1
    action=dict(format='vpp2_wan21_action_v1',step=30000,resolved_config=OmegaConf.to_container(cfg),
        action_expert={'action_encoder.weight':torch.empty(512,7,device='meta')},
        proprio_encoder={'weight':torch.empty(4096,8,device='meta')})
    monkeypatch.setattr(torch,'load',lambda path,**kwargs:action if path=='action' else dict(step=10000,dit={'test':True}))
    evaluation=OmegaConf.create(dict(action_checkpoint='action',video_checkpoint='video',action_step=30000))
    if accepted:
        assert validate_checkpoint(evaluation)['action_step']==30000
    else:
        with pytest.raises(ValueError,match='global-batch256'):
            validate_checkpoint(evaluation)
