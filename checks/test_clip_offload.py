"""Check inference placement using the native CLIP preprocessing/transformer."""

import pytest
import torch
from torch import nn

from vpp2.models.wan21_14b import wan_video_clip
from vpp2.models.wan21_14b.vpp2 import VPP2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clip_offload_preserves_features_and_releases_on_error(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    transformer = wan_video_clip.VisionTransformer
    monkeypatch.setattr(
        wan_video_clip, "VisionTransformer",
        lambda: transformer(image_size=28, patch_size=14, dim=32, out_dim=16,
                            num_heads=4, num_layers=3),
    )
    model = VPP2.__new__(VPP2)
    nn.Module.__init__(model)
    model.device = torch.device(device)
    model.torch_dtype = torch.bfloat16
    model.clip_encoder = wan_video_clip.WanImageEncoderCLIP()
    model.vae = nn.Linear(1, 1)
    video = torch.randn(2, 3, 2, 32, 48, device=device)
    reference = model._encode_clip_fea_from_video(video)
    model.offload_clip = True
    for _ in range(2):
        actual = model._encode_clip_fea_from_video(video)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual.device.type == device
        assert actual.dtype == torch.bfloat16
        assert all(p.device.type == "cpu" for p in model.clip_encoder.parameters())
        assert not any(p.requires_grad for p in model.clip_encoder.parameters())
        assert model.vae.weight.device.type == device
        assert model.vae.weight.dtype == torch.float32

    def fail(*args):
        raise RuntimeError("test encoder failure")

    handle = model.clip_encoder.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="test encoder failure"):
        model._encode_clip_fea_from_video(video)
    handle.remove()
    assert all(p.device.type == "cpu" for p in model.clip_encoder.parameters())
