import torch
from torch import nn
from torch.nn import functional as F
from .blocks import BackwardWarp


def local_corr(a, b, r=4):
    if a.shape != b.shape:
        raise ValueError('local correlation feature shapes must match')
    height, width = a.shape[-2:]
    bp = F.pad(b, (r, r, r, r))
    return torch.cat([(a*bp[:, :, dy:dy+height, dx:dx+width]).mean(1, keepdim=True)
                       for dy in range(2*r+1) for dx in range(2*r+1)], 1)


class FlowCorrelationRefiner(nn.Module):
    def __init__(self, radius=4, legacy_warp=False):
        super().__init__()
        self.radius = radius
        self.warp = BackwardWarp(legacy_warp)
        self.anchor = nn.Sequential(nn.Conv2d(512, 128, 1), nn.LeakyReLU(.1),
                                    nn.Conv2d(128, 128, 3, padding=1))
        count = (2*radius+1)**2
        self.decoder = nn.Sequential(nn.Conv2d(count+128+2, 128, 3, padding=1), nn.LeakyReLU(.1),
                                     nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(.1),
                                     nn.Conv2d(64, 2, 3, padding=1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def normalize(feature):
        mean = feature.mean(1, keepdim=True)
        variance = feature.var(1, keepdim=True, unbiased=False)
        return (feature-mean)/(variance+1e-6).sqrt()

    def forward(self, q, if0_4, if1_4):
        anchor = self.anchor(torch.cat((if0_4, if1_4, q['ctx0'], q['ctx1']), 1))
        norm_anchor = self.normalize(anchor)
        result = dict(q)
        for key, feature in (('f_t0', if0_4), ('f_t1', if1_4)):
            warped = self.normalize(self.warp(feature, q[key]))
            corr = local_corr(norm_anchor, warped, self.radius)
            result[key] = q[key] + self.decoder(torch.cat((corr, anchor, q[key]), 1))
        return result

