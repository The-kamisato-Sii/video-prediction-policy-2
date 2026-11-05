"""MotionCrafter's exact codec architecture, with no pretrained motion weights."""
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .codec import MotionCodec
from .views import DEFAULT_MOTION_CAMERAS, motion_cameras


class ReconstructionCodec(MotionCodec):
    decoder_names = ("decoder", "post_quant_conv", "decoder_2", "post_quant_conv_2")
    encoder_names = ("encoder", "quant_conv", "encoder_2", "quant_conv_2")

    def __init__(self, directory, frame_batch=4, cameras=DEFAULT_MOTION_CAMERAS):
        # Match MotionCodec module names, shapes and initialization RNG draws.
        # In particular, NEVER read diffusion_pytorch_model.safetensors here.
        nn.Module.__init__(self)
        self.motion_cameras = motion_cameras(cameras)
        self.num_views = len(self.motion_cameras)
        self.latent_channels = 8 * self.num_views
        from diffusers import AutoencoderKL
        config = json.loads((Path(directory) / "config.json").read_text())
        base = AutoencoderKL.from_config(config)
        self.encoder, self.quant_conv = base.encoder, base.quant_conv
        self.decoder, self.post_quant_conv = base.decoder, base.post_quant_conv
        self.encoder_2, self.quant_conv_2 = deepcopy(base.encoder), deepcopy(base.quant_conv)
        self.decoder_2, self.post_quant_conv_2 = deepcopy(base.decoder), deepcopy(base.post_quant_conv)
        c = self.decoder_2.conv_in
        self.decoder_2.conv_in = nn.Conv2d(8, c.out_channels, c.kernel_size, c.stride, c.padding)
        self.scaling_factor = float(config["scaling_factor"])
        self.frame_batch = int(frame_batch)
        if self.frame_batch < 1:
            raise ValueError("codec_frame_batch must be positive")
        self.requires_grad_(False)
        for name in self.decoder_names:
            getattr(self, name).requires_grad_(True)
        self.train()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # Retain unused encoders so the model's parameter inventory is unchanged.
        for name in self.encoder_names:
            getattr(self, name).eval()
        return self

    def encode(self, maps):
        raise RuntimeError("Direct reconstruction must never encode its GT labels")

    def _decode_frames(self, part):
        geom, motion = part[:, :4], part[:, 4:]
        g = self.decoder(self.post_quant_conv(geom))
        f = self.decoder_2(torch.cat([self.post_quant_conv_2(motion), geom], dim=1)) / 10
        return torch.cat([g, f], dim=1)

    def decode(self, latent):
        b, c, t, h, w = latent.shape
        if (c, t, h, w) != (self.latent_channels, 8, 28, 28):
            raise ValueError(f"Bad joint latent shape {tuple(latent.shape)}")
        z = latent.reshape(b, self.num_views, 8, t, h, w).permute(0, 1, 3, 2, 4, 5).reshape(-1, 8, h, w)
        output = []
        for part in (z / self.scaling_factor).split(self.frame_batch):
            # Recompute decoder activations during backward; otherwise all16B
            # 224x224 frame graphs remain live. This adds no model parameters.
            if torch.is_grad_enabled():
                decoded = checkpoint(self._decode_frames, part, use_reentrant=False)
            else:
                decoded = self._decode_frames(part)
            output.append(decoded)
        return torch.cat(output).reshape(b, self.num_views, t, 6, 224, 224).permute(0, 1, 3, 2, 4, 5).contiguous()
