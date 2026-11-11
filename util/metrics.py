import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio,structural_similarity


class ImageMetrics:
    def __init__(self,device='cpu',use_lpips=True,legacy_lpips=False):
        self.legacy_lpips=legacy_lpips
        self.lpips=None
        if use_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError('install lpips/torchvision or explicitly use --skip_lpips') from exc
            self.lpips=lpips.LPIPS(net='alex').to(device).eval()

    @torch.no_grad()
    def __call__(self,pred,target):
        pred=pred.clamp(0,1)
        p=(pred[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        y=(target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        minimum=min(p.shape[:2]);window=min(7,minimum if minimum%2 else minimum-1)
        if window<3:raise ValueError('metric images must be at least 3 pixels in each spatial dimension')
        result=dict(psnr=float(peak_signal_noise_ratio(y,p,data_range=255)),
                    ssim=float(structural_similarity(y,p,data_range=255,channel_axis=2,win_size=window)))
        if self.lpips is not None:
            result['lpips']=float(self.lpips(pred,target,normalize=not self.legacy_lpips).mean())
        return result

