"""Frozen official MotionCrafter codec, loaded without remote code or SD downloads."""
from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from .views import DEFAULT_MOTION_CAMERAS, motion_cameras


class MotionCodec(nn.Module):
    def __init__(self, directory, frame_batch=4, cameras=DEFAULT_MOTION_CAMERAS):
        super().__init__()
        self.motion_cameras = motion_cameras(cameras)
        self.num_views = len(self.motion_cameras)
        self.latent_channels = 8 * self.num_views
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        base = AutoencoderKL.from_config(config)
        self.encoder, self.quant_conv = base.encoder, base.quant_conv
        self.decoder, self.post_quant_conv = base.decoder, base.post_quant_conv
        self.encoder_2, self.quant_conv_2 = deepcopy(base.encoder), deepcopy(base.quant_conv)
        self.decoder_2, self.post_quant_conv_2 = deepcopy(base.decoder), deepcopy(base.post_quant_conv)
        c = self.decoder_2.conv_in
        self.decoder_2.conv_in = nn.Conv2d(8, c.out_channels, c.kernel_size, c.stride, c.padding)
        state = load_file(str(directory / "diffusion_pytorch_model.safetensors"))
        self.load_state_dict(state, strict=True)
        self.scaling_factor = float(config["scaling_factor"])
        self.frame_batch = int(frame_batch)
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, maps):
        # B,V,6,T,H,W -> B,V,8,T,28,28 -> B,8*V,T,28,28.
        b, v, c, t, h, w = maps.shape
        if (v, c, t, h, w) != (self.num_views, 6, 8, 224, 224):
            raise ValueError(f"Expected configured 8-frame224 motion views, got{tuple(maps.shape)}")
        x = maps.permute(0, 1, 3, 2, 4, 5).reshape(-1, 6, h, w)
        output = []
        for part in x.split(self.frame_batch):
            g = self.quant_conv(self.encoder(part[:, :3])).chunk(2, dim=1)[0]
            f = self.quant_conv_2(self.encoder_2(part[:, 3:] * 10)).chunk(2, dim=1)[0]
            output.append(torch.cat([g, f], dim=1) * self.scaling_factor)
        z = torch.cat(output).reshape(b, v, t, 8, 28, 28)
        return z.permute(0, 1, 3, 2, 4, 5).reshape(b, self.latent_channels, t, 28, 28).contiguous()

    def decode(self, latent):
        # Frozen parameters, but retain gradients to latent for optional3D loss.
        b, c, t, h, w = latent.shape
        if (c, t, h, w) != (self.latent_channels, 8, 28, 28):
            raise ValueError(f"Bad joint latent shape{tuple(latent.shape)}")
        z = latent.reshape(b, self.num_views, 8, t, h, w).permute(0, 1, 3, 2, 4, 5).reshape(-1, 8, h, w)
        output = []
        for part in (z / self.scaling_factor).split(self.frame_batch):
            geom, motion = part[:, :4], part[:, 4:]
            g = self.decoder(self.post_quant_conv(geom))
            f = self.decoder_2(torch.cat([self.post_quant_conv_2(motion), geom], dim=1)) / 10
            output.append(torch.cat([g, f], dim=1))
        return torch.cat(output).reshape(b, self.num_views, t, 6, 224, 224).permute(0, 1, 3, 2, 4, 5).contiguous()
