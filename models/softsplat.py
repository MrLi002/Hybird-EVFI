"""OF-AF splatting API with a differentiable PyTorch reference backend.

CUDA/CuPy implementation remains unchanged in softsplat_cuda.py. CPU execution
and explicit --splat_backend torch use bilinear scatter_add with the same epsilon.
"""
import warnings
import torch

_CUDA_IMPL = None
_CUDA_FAILED = False


def torch_softsplat(tenIn, tenFlow, tenMetric=None, strMode='avg'):
    mode = strMode.split('-')[0]
    if mode not in ('sum', 'avg', 'linear', 'soft'):
        raise ValueError(f'unsupported splat mode {strMode}')
    if tenFlow.shape != (tenIn.shape[0], 2, *tenIn.shape[-2:]):
        raise ValueError('source and flow shapes must match')
    original_dtype = tenIn.dtype
    dtype = torch.float64 if tenIn.dtype == torch.float64 else torch.float32
    src, flow = tenIn.to(dtype), tenFlow.to(dtype)
    batch, channels, height, width = src.shape
    if mode == 'sum':
        if tenMetric is not None:
            raise ValueError('sum splatting takes no metric')
        payload = src
    else:
        if mode in ('linear', 'soft'):
            if tenMetric is None:
                raise ValueError('weighted splatting needs a source-grid metric')
            weight = tenMetric.to(dtype)
            weight = weight.exp() if mode == 'soft' else weight
        else:
            if tenMetric is not None:
                raise ValueError('avg splatting takes no metric')
            weight = torch.ones_like(src[:, :1])
        payload = torch.cat((src*weight, weight), 1)
    yy, xx = torch.meshgrid(torch.arange(height, device=src.device, dtype=dtype),
                            torch.arange(width, device=src.device, dtype=dtype), indexing='ij')
    x, y = xx[None]+flow[:, 0], yy[None]+flow[:, 1]
    left, top = x.floor(), y.floor()
    out = src.new_zeros(batch, payload.shape[1], height*width)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        xi, yi = left+dx, top+dy
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        bilinear = (1-(x-xi).abs()) * (1-(y-yi).abs())
        bilinear = torch.where(valid, bilinear, torch.zeros_like(bilinear))
        index = (yi.clamp(0, height-1).long()*width + xi.clamp(0, width-1).long()).flatten(1)
        out.scatter_add_(2, index[:, None].expand(-1, payload.shape[1], -1),
                         (payload*bilinear[:, None]).flatten(2))
    out = out.view(batch, payload.shape[1], height, width)
    if mode != 'sum':
        den = out[:, -1:]
        suffix = strMode.split('-')[1] if '-' in strMode else 'addeps'
        if suffix == 'addeps':
            den = den + 1e-7
        elif suffix == 'zeroeps':
            den = torch.where(den == 0, torch.ones_like(den), den)
        elif suffix == 'clipeps':
            den = den.clamp_min(1e-7)
        else:
            raise ValueError(f'unknown normalization {suffix}')
        out = out[:, :-1]/den
    return out.to(original_dtype)


def softsplat(tenIn, tenFlow, tenMetric=None, strMode='avg', backend='auto'):
    global _CUDA_IMPL, _CUDA_FAILED
    if backend not in ('auto', 'torch', 'cupy'):
        raise ValueError('backend must be auto, torch or cupy')
    if backend == 'cupy' and not tenIn.is_cuda:
        raise ValueError('CuPy splatting requires CUDA tensors')
    if tenIn.is_cuda and backend != 'torch':
        if _CUDA_IMPL is None and not _CUDA_FAILED:
            try:
                from .softsplat_cuda import softsplat as cuda_impl
                _CUDA_IMPL = cuda_impl
            except ImportError:
                _CUDA_FAILED = True
                if backend == 'auto':
                    warnings.warn('CuPy unavailable; using slower PyTorch splatting.', stacklevel=2)
        if _CUDA_IMPL is not None:
            dtype = tenIn.dtype
            return _CUDA_IMPL(tenIn.float(), tenFlow.float(),
                              None if tenMetric is None else tenMetric.float(), strMode).to(dtype)
        if backend == 'cupy':
            raise ImportError('install a CuPy package matching your CUDA runtime')
    return torch_softsplat(tenIn, tenFlow, tenMetric, strMode)

