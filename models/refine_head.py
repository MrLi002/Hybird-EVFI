import torch
from torch import nn
from torch.nn import functional as F
from .blocks import BackwardWarp, flow_to_scale
from .cbm_modules import Transformer, frame_encoder
from .cbm_submodules import conv3x3, conv3x3_leaky_relu
from .basr import DetailsSup, make_basr_masks


class RefinementTransformer(Transformer):
    def __init__(self, unit_dim=32, extras=3, basr=False, heads=(4, 2), chunk_size=256, support_ratio=1.0):
        U = 2*unit_dim
        super().__init__(U)
        self.use_basr = basr
        self.enc_conv0 = conv3x3_leaky_relu(U+6+extras, U)
        self.build_kv0_warp = conv3x3_leaky_relu(3*U+6+extras, 4*U)
        self.build_kv1_warp = conv3x3_leaky_relu(int(1.5*U)+6+extras, 2*U)
        self.build_kv2_warp = conv3x3_leaky_relu(int(.75*U)+6+extras, U)
        if basr:
            self.conv0 = conv3x3(4*U, U//2)
            self.conv1 = conv3x3(2*U, U//4)
            self.refine_block0 = DetailsSup(8, U//2, U//2, heads[0], chunk_size, support_ratio)
            self.refine_block1 = DetailsSup(16, U//4, U//4, heads[1], chunk_size, support_ratio)
        for head in (self.last_conv0, self.last_conv1, self.last_conv2):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, event_feature, frame_feature, warped_feature, basr_data=None):
        queries = self.build_qurey(event_feature, frame_feature, warped_feature)
        features = []
        up = None
        for level, s in enumerate((2, 1, 0)):
            query = queries[s]
            if level:
                query = getattr(self, f'conv_after_body{level-1}')(torch.cat((query, up), 1))
            syn = getattr(self, f'build_kv{level}_syn')(torch.cat((frame_feature[s], event_feature[s]), 1))
            wrp = getattr(self, f'build_kv{level}_warp')(torch.cat((warped_feature[s], event_feature[s]), 1))
            out = getattr(self, f'decoder{level+1}_1')(query, syn, wrp)
            out = getattr(self, f'decoder{level+1}_2')(out)
            features.append(out)
            if level < 2:
                up = getattr(self, f'upsample{level}')(out)
                if self.use_basr:
                    if basr_data is None:
                        raise ValueError('BASR was enabled but its reference features/masks are missing')
                    query_small = getattr(self, f'conv{level}')(out)
                    up = getattr(self, f'refine_block{level}')(
                        query_small, basr_data['ref0'][s], basr_data['ref1'][s], up, basr_data['masks'])
        return [self.last_conv2(features[2]), self.last_conv1(features[1]), self.last_conv0(features[0])]


class RefineHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_in_refine, self.use_basr = config.mask_in_refine, config.basr
        self.encoder_f = frame_encoder(3, config.unit_dim//4)
        self.warp = BackwardWarp(config.legacy_warp)
        self.transformer = RefinementTransformer(config.unit_dim, 7 if config.mask_in_refine else 3,
            config.basr, config.basr_heads, config.basr_chunk_size, config.basr_support_ratio)

    @staticmethod
    def image_pyramid(image):
        return [image] + [F.interpolate(image, scale_factor=.5**s, mode='bilinear', align_corners=False)
                          for s in (1, 2)]

    def encode_frames(self, im0, im1):
        return dict(f0=self.encoder_f(im0), f1=self.encoder_f(im1),
                    im0=self.image_pyramid(im0), im1=self.image_pyramid(im1))

    def extras_at_scale(self, w, s):
        size = (w['fuse'].shape[-2]//2**s, w['fuse'].shape[-1]//2**s)
        fuse = F.interpolate(w['fuse'], size=size, mode='bilinear', align_corners=False)
        if not self.mask_in_refine:
            return fuse
        hole0 = F.interpolate(w['a']['hole'], size=size, mode='nearest')
        hole1 = F.interpolate(w['b']['hole'], size=size, mode='nearest')
        diff0 = F.adaptive_avg_pool2d(w['a']['diff'], size)
        diff1 = F.adaptive_avg_pool2d(w['b']['diff'], size)
        return torch.cat((fuse, hole0, hole1, diff0, diff1), 1)

    def forward(self, pyr, q, w, ev):
        frame, warped, fuse_ms, refs0, refs1 = [], [], [], [], []
        for s in range(3):
            f0, f1 = pyr['f0'][s], pyr['f1'][s]
            w0 = self.warp(torch.cat((f0, pyr['im0'][s]), 1), flow_to_scale(q['f_t0'], s))
            w1 = self.warp(torch.cat((f1, pyr['im1'][s]), 1), flow_to_scale(q['f_t1'], s))
            extra = self.extras_at_scale(w, s)
            frame.append(torch.cat((f0, f1), 1))
            warped.append(torch.cat((w0, w1, extra), 1))
            fuse_ms.append(extra[:, :3])
            refs0.append(w0[:, :f0.shape[1]])
            refs1.append(w1[:, :f1.shape[1]])
        data = None
        if self.use_basr:
            data = dict(ref0=refs0, ref1=refs1, masks=make_basr_masks(w))
        residuals = self.transformer(ev, frame, warped, basr_data=data)
        return [base+residual for base, residual in zip(fuse_ms, residuals)]

