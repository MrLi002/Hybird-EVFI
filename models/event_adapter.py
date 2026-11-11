import torch
from torch import nn
from .cbm_modules import event_encoder


class EventAdapter(nn.Module):
    """Full-interval voxel features conditioned on time, never hard sliced at t."""
    def __init__(self, unit_dim=32, ctx_ch=128, bins=128):
        super().__init__()
        u = unit_dim
        self.bins = bins
        self.enc = event_encoder(bins, u//2)
        self.proj = nn.ModuleList(nn.Conv2d(c, 2*c, 1) for c in (u//2, u, 2*u))
        self.time_proj = nn.ModuleList(nn.Conv2d(2, c, 1) for c in (u, 2*u, 4*u))
        self.fuse = nn.Conv2d(4*u+2*ctx_ch, 4*u, 1)

    def encode_events(self, event):
        if event.shape[1] != self.bins:
            raise ValueError('wrong number of input voxel bins')
        return [proj(feat) for proj, feat in zip(self.proj, self.enc(event))]

    def forward(self, global_feats, t, ctx0, ctx1):
        t = torch.as_tensor(t, device=ctx0.device, dtype=ctx0.dtype)
        code = torch.stack((t, 1-t)).reshape(1, 2, 1, 1).expand(ctx0.shape[0], -1, -1, -1)
        features = [feat+proj(code) for feat, proj in zip(global_feats, self.time_proj)]
        features[2] = self.fuse(torch.cat((features[2], ctx0, ctx1), 1))
        return features

