"""OF-AF bidirectional warping; learned parameter names match the upstream model."""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ResBlock, ResBlockIF, BackwardWarp, flow_to_scale
from .softsplat import softsplat


class RefBlock(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(base_channel*4, base_channel*2, 3, 1, 1)
        self.conv1 = nn.Conv2d(base_channel*4+1, base_channel*2, 3, 1, 1)
        self.res = ResBlock(base_channel*8+1, base_channel*4+1)

    def forward(self, mask, feature):
        mask_feature_s = self.conv0(mask * feature)
        mask_feature_w = self.conv1(torch.cat([feature, mask], dim=1))
        out = self.res(torch.cat([mask_feature_s, mask_feature_w, feature, mask], dim=1))
        return mask + out[:, :1], out[:, 1:]


class MaskGuideNet(nn.Module):
    def __init__(self, base_channel=32, legacy_warp=False, backend='auto', metric=False):
        super().__init__()
        self.bwarp = BackwardWarp(legacy_warp)
        self.backend = backend
        self.refblock0 = RefBlock(base_channel)
        self.refblock1 = RefBlock(base_channel)
        self.res0 = ResBlock(base_channel*8+3, base_channel*4)
        self.conv0 = nn.Conv2d(7, base_channel*4, 3, 1, 1)
        self.conv1 = nn.Conv2d(base_channel*8, 3, 3, 1, 1)
        if metric:
            self.metric_head = nn.Conv2d(4*base_channel+2, 1, 3, 1, 1)
            nn.init.zeros_(self.metric_head.weight)
            nn.init.zeros_(self.metric_head.bias)

    def splat(self, src, flow, metric=None, mode='avg'):
        return softsplat(src, flow, metric, mode, backend=self.backend)

    def forward(self, im0, im1, flow0t, flowt0, ctxt0, ctxt1, src_feat=None):
        batch, _, h, w = flow0t.shape
        Flowt0, Flow0t = flow_to_scale(flowt0, 0), flow_to_scale(flow0t, 0)
        metric = None
        if hasattr(self, 'metric_head'):
            if src_feat is None:
                raise ValueError('splat metric needs source-grid image features')
            metric = self.metric_head(torch.cat((src_feat, flow0t), 1))
            metric = F.interpolate(metric, size=im0.shape[-2:], mode='bilinear', align_corners=False)
            metric = metric.clamp(-20, 20)
        im0t = self.splat(im0, Flow0t, metric, 'soft' if metric is not None else 'avg')
        imt0, in_bounds = self.bwarp(im0, Flowt0, return_valid=True)
        ones = im0.new_ones(batch, 1, h, w)
        # Preserve upstream internal mask arithmetic, including its normalization epsilon.
        hole_raw = 1-self.splat(ones, flow0t)
        diff_map = (imt0-im0t).abs().sum(1, keepdim=True)
        diff_mask = (diff_map > .1).to(im0.dtype)
        coverage_q = self.splat(ones, flow0t, mode='sum')
        hole_binary = (coverage_q <= 1e-6).to(im0.dtype)
        coverage_full = self.splat(torch.ones_like(im0[:, :1]), Flow0t, mode='sum')

        hole_mask, ref0 = self.refblock0(hole_raw, ctxt0)
        hole_mask, ref1 = self.refblock1(hole_mask, ctxt1)
        im1_4 = F.interpolate(im1, scale_factor=1/4, mode='bilinear', align_corners=False)
        ref = self.res0(torch.cat((ref0, ref1, im1_4), 1))
        Ref = F.interpolate(ref, scale_factor=4, mode='bilinear', align_corners=False)
        Mask0t = F.interpolate(hole_mask, scale_factor=4, mode='bilinear', align_corners=False) + diff_mask
        ref_source = self.conv0(torch.cat((imt0*Mask0t, imt0, Mask0t), 1))
        corrected = imt0 + self.conv1(torch.cat((Ref, ref_source), 1))
        return dict(imt=corrected, im_bwd=imt0, im_fwd=im0t, mask=Mask0t,
                    hole=hole_binary, hole_raw=hole_raw, diff=diff_mask,
                    coverage=coverage_full, in_bounds=in_bounds)


class BiW(nn.Module):
    def __init__(self, base_channel=32, legacy_warp=False, backend='auto', metric=False):
        super().__init__()
        self.fuse_warp = MaskGuideNet(base_channel, legacy_warp, backend, metric)
        self.channel_squeeze0 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.channel_squeeze1 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.refine_blocks = ResBlockIF(3+128, 64)
        self.conv_out = nn.Conv2d(64, 3, 1, 1, bias=True)

    def forward(self, im0, im1, t, q, src_feat0=None, src_feat1=None):
        a = self.fuse_warp(im0, im1, q['f_0t'], q['f_t0'], q['ctx0'], q['ctx1'], src_feat0)
        b = self.fuse_warp(im1, im0, q['f_1t'], q['f_t1'], q['ctx1'], q['ctx0'], src_feat1)
        blend = torch.softmax(torch.cat(((1-a['mask'])*(1-t), (1-b['mask'])*t), 1), 1)[:, :1]
        return dict(fuse=a['imt']*blend+(1-blend)*b['imt'], blend=blend, a=a, b=b)

    def refine_resblock(self, w, q, if0_1, if1_1):
        feature0 = F.interpolate(self.channel_squeeze0(q['ctx0']), scale_factor=4,
                                 mode='bilinear', align_corners=False)
        feature1 = F.interpolate(self.channel_squeeze1(q['ctx1']), scale_factor=4,
                                 mode='bilinear', align_corners=False)
        inputs = torch.cat((feature0, feature1, w['fuse'], if0_1, if1_1), 1)
        return self.conv_out(self.refine_blocks(inputs)) + w['fuse']

