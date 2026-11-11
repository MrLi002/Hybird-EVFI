"""OF-AF layout: estimate interval motion once, query/warp/refine at each t."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .encoder import EventsEncoder, ImageEncoder
from .BiFEB import BidirFlow
from .BiW import BiW
from .blocks import BackwardWarp
from .query import FlowQuery
from .event_adapter import EventAdapter
from .refine_head import RefineHead
from .flow_corr import FlowCorrelationRefiner
from .basr import make_basr_masks, effective_repair_gate


class HybridModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.config = cfg = ModelConfig.from_args(args)
        self.base_channel = 32
        self.voxel_bins, self.nb_of_flow = cfg.voxel_bins, cfg.nb_of_flow
        self.bins_per_flow = cfg.voxel_bins//cfg.nb_of_flow
        self.refine = cfg.refine
        self.events_encoder = EventsEncoder(cfg.voxel_bins, 32, self.bins_per_flow)
        self.im_encoder = ImageEncoder(3, 32)
        self.bidir_flow0, self.bidir_flow1 = BidirFlow(32), BidirFlow(32)
        self.bidir_flow0.bwarp.legacy = cfg.legacy_warp
        self.bidir_flow1.bwarp.legacy = cfg.legacy_warp
        self.query = FlowQuery(cfg.nb_of_flow, cfg.legacy_query)
        self.biw = BiW(32, cfg.legacy_warp, cfg.splat_backend, cfg.splat_metric)
        self.flow_corr = FlowCorrelationRefiner(cfg.correlation_radius, cfg.legacy_warp) if cfg.flow_corr else None
        if cfg.refine == 'transformer':
            self.event_adapter = EventAdapter(cfg.unit_dim, 128, cfg.voxel_bins)
            self.refine_head = RefineHead(cfg)

    @property
    def size_multiple(self):
        return 32 if self.config.basr else 4

    def run_bifeb(self, if0, if1, events):
        zero = if0.new_zeros(if0.shape[0], 2, *if0.shape[-2:])
        ft0, ft1, context0, context1 = if0, if1, if0, if1
        motion0, motion1 = torch.zeros_like(if0), torch.zeros_like(if1)
        flowt0, flowt1 = zero, zero
        knots = dict(flows0t=[zero], flows1t=[zero], flowst0=[zero], flowst1=[zero],
                     ctxst0=[if0], ctxst1=[if1])
        def step(module, *inputs):
            if self.training and self.config.checkpoint_bifeb:
                return checkpoint(module, *inputs, use_reentrant=False)
            return module(*inputs)
        for index in range(self.nb_of_flow):
            ft0, motion0, forward0, flowt0, context0 = step(
                self.bidir_flow0, if0, ft0, flowt0, motion0, context0, events[:, index])
            ft1, motion1, forward1, flowt1, context1 = step(
                self.bidir_flow1, if1, ft1, flowt1, motion1, context1, events[:, self.nb_of_flow-index-1])
            knots['flows0t'].append(forward0)
            knots['flowst0'].append(flowt0)
            knots['ctxst0'].append(context0)
            knots['flows1t'].insert(0, forward1)
            knots['flowst1'].insert(0, flowt1)
            knots['ctxst1'].insert(0, context1)
        return knots

    def encode_pair(self, im0, im1, event):
        if im0.shape != im1.shape or im0.ndim != 4 or im0.shape[1] != 3:
            raise ValueError('keyframes must have identical (B,3,H,W) shapes')
        if event.shape != (im0.shape[0], self.voxel_bins, *im0.shape[-2:]):
            raise ValueError('event shape does not match configured bins/keyframes')
        if any(size % self.size_multiple for size in im0.shape[-2:]):
            raise ValueError(f'pad inputs to multiples of {self.size_multiple} before calling model')
        feature0, quarter0 = self.im_encoder(im0)
        feature1, quarter1 = self.im_encoder(im1)
        knots = self.run_bifeb(quarter0, quarter1, self.events_encoder(im0, im1, event))
        cached = dict(im0=im0, im1=im1, if0_1=feature0, if1_1=feature1,
                      if0_4=quarter0, if1_4=quarter1, knots=knots)
        if self.refine == 'transformer':
            cached['pyr'] = self.refine_head.encode_frames(im0, im1)
            cached['event_global'] = self.event_adapter.encode_events(event)
        return cached

    def render_time(self, cached, t, return_aux=False):
        q_raw = self.query(cached['knots'], t)
        q = self.flow_corr(q_raw, cached['if0_4'], cached['if1_4']) if self.flow_corr is not None else q_raw
        w = self.biw(cached['im0'], cached['im1'], t, q, cached['if0_4'], cached['if1_4'])
        if self.refine == 'resblock':
            ms = [self.biw.refine_resblock(w, q, cached['if0_1'], cached['if1_1'])]
        else:
            ev = self.event_adapter(cached['event_global'], t, q['ctx0'], q['ctx1'])
            if self.training and self.config.checkpoint_refine:
                ms = checkpoint(self.refine_head, cached['pyr'], q, w, ev, use_reentrant=False)
            else:
                ms = self.refine_head(cached['pyr'], q, w, ev)
        if return_aux:
            masks=make_basr_masks(w) if self.config.basr else None
            if masks is not None:
                masks['effective_gate']=effective_repair_gate(masks,self.config.basr_support_ratio)
            return ms, dict(q_raw=q_raw, q=q, w=w,
                             basr_masks=masks)
        return ms[0]

    def forward(self, im0, im1, event, interp_times, return_aux=False):
        if interp_times.ndim != 2 or interp_times.shape[0] != im0.shape[0] or interp_times.shape[1] == 0:
            raise ValueError('interp_times must have nonempty shape (B,K)')
        if not torch.allclose(interp_times, interp_times[:1].expand_as(interp_times)):
            raise ValueError('samples in a batch must share the same interpolation times')
        if not torch.isfinite(interp_times).all() or not ((interp_times>0)&(interp_times<1)).all():
            raise ValueError('interpolation times must be finite and in (0,1)')
        cached = self.encode_pair(im0, im1, event)
        predictions, multiscale, auxiliary = [], [], []
        for t in interp_times[0]:
            if return_aux:
                ms, aux = self.render_time(cached, t, True)
                predictions.append(ms[0]); multiscale.append(ms); auxiliary.append(aux)
            else:
                predictions.append(self.render_time(cached, t))
        prediction = torch.stack(predictions, 1)
        return dict(pred=prediction, pred_ms=multiscale, aux=auxiliary) if return_aux else prediction


MyModel = HybridModel

