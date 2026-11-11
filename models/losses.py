import torch
from torch import nn
from torch.nn import functional as F
from .blocks import BackwardWarp, flow_to_scale


def charbonnier(pred, target, eps=1e-3):
    return ((pred-target).square()+eps**2).sqrt().mean()


def masked_charbonnier(pred, target, valid, eps=1e-3):
    if pred.shape != target.shape or valid.shape != (pred.shape[0], 1, *pred.shape[-2:]):
        raise ValueError('masked loss requires matching images and a single-channel mask')
    valid = valid.detach().to(pred.dtype)
    error = ((pred-target).square()+eps**2).sqrt()
    return (error*valid).sum() / (valid.sum()*pred.shape[1]).clamp_min(1e-6)


def cal_grad2_error(flow, image, beta=1.0):
    def gradient(x):
        return x[:, :, :, 1:]-x[:, :, :, :-1], x[:, :, 1:, :]-x[:, :, :-1, :]
    if min(flow.shape[-2:]) < 3:
        return flow.sum()*0
    ix, iy = gradient(image)
    wx = torch.exp(-10*ix.abs().mean(1, keepdim=True))
    wy = torch.exp(-10*iy.abs().mean(1, keepdim=True))
    dx, dy = gradient(flow)
    dxx, _ = gradient(dx)
    _, dyy = gradient(dy)
    return beta*((wx[:, :, :, 1:]*dxx.abs()).mean()+(wy[:, :, 1:, :]*dyy.abs()).mean())/2


def ssim_map(pred, target, window_size=11):
    if pred.shape != target.shape or min(pred.shape[-2:]) < window_size:
        raise ValueError('SSIM images must match and be at least window_size pixels wide/high')
    dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
    x, y = pred.to(dtype), target.to(dtype)
    coords = torch.arange(window_size, device=x.device, dtype=dtype)-(window_size//2)
    g = torch.exp(-coords.square()/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:, None]*g[None, :])[None, None].expand(x.shape[1], 1, -1, -1)
    conv = lambda z: F.conv2d(z, kernel, groups=x.shape[1])
    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x*x)-mx*mx, conv(y*y)-my*my, conv(x*y)-mx*my
    return ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))


def masked_ssim_loss(pred, target, region, window_size=11):
    if region.shape != (pred.shape[0], 1, *pred.shape[-2:]):
        raise ValueError('SSIM region must be single-channel on the target grid')
    scores = ssim_map(pred, target, window_size)
    r = window_size//2
    center = region[:, :, r:region.shape[-2]-r, r:region.shape[-1]-r].detach().to(scores.dtype)
    return (center*(1-scores)).sum() / (center.sum()*pred.shape[1]).clamp_min(1e-6)


class HybridLoss(nn.Module):
    def __init__(self, w_lpips=.1, w_flow=0., w_ps=0., legacy_warp=False, legacy_lpips=False,
                 lpips_net='vgg', lpips_model=None):
        super().__init__()
        self.w_lpips, self.w_flow, self.w_ps = w_lpips, w_flow, w_ps
        self.legacy_lpips = legacy_lpips
        self.warp = BackwardWarp(legacy_warp)
        self.lpips = None
        if w_lpips:
            if lpips_model is None:
                try:
                    import lpips
                except ImportError as exc:
                    raise ImportError('install lpips/torchvision or explicitly use --w_lpips 0') from exc
                lpips_model = lpips.LPIPS(net=lpips_net)
            self.lpips = lpips_model.eval()
            for parameter in self.lpips.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.lpips is not None:
            self.lpips.eval()
        return self

    def flow_loss(self, q, w, im0, im1, gt):
        f0, f1 = flow_to_scale(q['f_0t'], 0), flow_to_scale(q['f_1t'], 0)
        b0, b1 = flow_to_scale(q['f_t0'], 0), flow_to_scale(q['f_t1'], 0)
        v0 = (w['a']['coverage']>1e-6).to(gt.dtype).detach()
        v1 = (w['b']['coverage']>1e-6).to(gt.dtype).detach()
        photo = (masked_charbonnier(self.warp(im0, b0), gt, v0*w['a']['in_bounds'])
                 + masked_charbonnier(self.warp(im1, b1), gt, v1*w['b']['in_bounds'])
                 + masked_charbonnier(w['a']['im_fwd'], gt, v0)
                 + masked_charbonnier(w['b']['im_fwd'], gt, v1))
        smooth = (cal_grad2_error(f0/20, im0)+cal_grad2_error(f1/20, im1)
                  + cal_grad2_error(b0/20, gt)+cal_grad2_error(b1/20, gt))
        return photo+10*smooth

    def forward(self, outputs, targets, im0, im1):
        if outputs['pred'].shape != targets.shape:
            raise ValueError('targets must have shape (B,K,3,H,W) matching predictions')
        count = targets.shape[1]
        loss_img = outputs['pred'].sum()*0
        for j, scales in enumerate(outputs['pred_ms']):
            gt = targets[:, j]
            for scale, (pred, weight) in enumerate(zip(scales, (1., .1, .1))):
                if scale:
                    gt = F.avg_pool2d(gt, 2, 2)
                loss_img = loss_img + weight*charbonnier(pred, gt)/count
        flat_pred = outputs['pred'].flatten(0, 1)
        flat_gt = targets.flatten(0, 1)
        zero = loss_img*0
        loss_lpips = self.lpips(flat_pred.float(), flat_gt.float(), normalize=not self.legacy_lpips).mean() if self.lpips is not None else zero
        loss_flow, loss_ps = zero, zero
        for j, auxiliary in enumerate(outputs['aux']):
            if self.w_flow:
                loss_flow = loss_flow+self.flow_loss(auxiliary['q'], auxiliary['w'], im0, im1, targets[:, j])/count
            if self.w_ps:
                masks = auxiliary['basr_masks']
                if masks is None:
                    raise ValueError('regional SSIM needs BASR masks')
                region=masks.get('effective_gate',masks['repair_gate'])
                loss_ps = loss_ps+masked_ssim_loss(outputs['pred'][:, j], targets[:, j], region)/count
        total = loss_img+self.w_lpips*loss_lpips+self.w_flow*loss_flow+self.w_ps*loss_ps
        return total, dict(img=loss_img.detach(), lpips=loss_lpips.detach(),
                           flow=loss_flow.detach(), ps=loss_ps.detach(), total=total.detach())

