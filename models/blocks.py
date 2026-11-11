import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_)
        return x_res + x_


class ResBlockIF(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )
        self.conv_block1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_) + x_
        x_res1 = self.conv_block1(x_res) + x_res
        return x_res1


def resize_flow(flow, size):
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = size
    if (old_h, old_w) == (new_h, new_w):
        return flow
    result = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
    scale = flow.new_tensor([new_w/old_w, new_h/old_h]).view(1, 2, 1, 1)
    return result * scale


def flow_to_scale(flow, s):
    if s not in (0, 1, 2):
        raise ValueError('scale must be 0, 1 or 2')
    factor = 4 // (2**s)
    return resize_flow(flow, (flow.shape[-2]*factor, flow.shape[-1]*factor))


class BackwardWarp(nn.Module):
    """Pixel-unit backward sampling; legacy reproduces OF-AF's half-pixel shift."""
    def __init__(self, legacy=False):
        super().__init__()
        self.legacy = legacy
        self.grid = {}

    def forward(self, img, flow, return_valid=False):
        if flow.shape != (img.shape[0], 2, *img.shape[-2:]):
            raise ValueError('flow must match image batch and spatial dimensions')
        height, width = img.shape[-2:]
        # float32 coordinate arithmetic also supports AMP inputs and avoids coarse half grids.
        dtype = torch.float64 if img.dtype == torch.float64 else torch.float32
        key = (height, width, img.device, dtype)
        if key not in self.grid:
            y, x = torch.meshgrid(torch.arange(height, device=img.device, dtype=dtype),
                                  torch.arange(width, device=img.device, dtype=dtype), indexing='ij')
            if len(self.grid) > 8:
                self.grid.clear()
            self.grid[key] = (x[None], y[None])
        x0, y0 = self.grid[key]
        x, y = x0 + flow[:, 0].to(dtype), y0 + flow[:, 1].to(dtype)
        if self.legacy:
            gx, gy = 2*(x/width - .5), 2*(y/height - .5)
        else:
            gx = 2*x/max(width-1, 1)-1 if width > 1 else x*0
            gy = 2*y/max(height-1, 1)-1 if height > 1 else y*0
        grid = torch.stack((gx, gy), -1)
        out = F.grid_sample(img.to(dtype), grid, mode='bilinear', padding_mode='zeros',
                            align_corners=not self.legacy).to(img.dtype)
        if not return_valid:
            return out
        sx, sy = (x-.5, y-.5) if self.legacy else (x, y)
        valid = ((sx >= 0) & (sx <= width-1) & (sy >= 0) & (sy <= height-1))[:, None]
        return out, valid.to(img.dtype)


def backwarp(img, flow, legacy=False, return_valid=False):
    return BackwardWarp(legacy)(img, flow, return_valid)



