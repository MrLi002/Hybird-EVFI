import torch
from torch import nn


class FlowQuery(nn.Module):
    def __init__(self, n_seg=16, legacy=False):
        super().__init__()
        if n_seg <= 0:
            raise ValueError('n_seg must be positive')
        self.n, self.legacy = n_seg, legacy

    def forward(self, knots, t):
        target = float(t)
        if not 0 < target < 1:
            raise ValueError('interpolation time must be strictly between 0 and 1')
        # Keep weights on-device; node selection is a small non-differentiable index operation.
        index = target * self.n
        i = min(int(index), self.n - 1)
        lam = index - i
        def blend(name, reverse=False):
            values = knots[name]
            if len(values) != self.n + 1:
                raise ValueError(f'{name} must contain {self.n+1} chronologically ordered nodes')
            a = lam if reverse else 1-lam
            return values[i] * a + values[i+1] * (1-a)
        return dict(f_0t=blend('flows0t'), f_1t=blend('flows1t', self.legacy),
                    f_t0=blend('flowst0', self.legacy), f_t1=blend('flowst1', self.legacy),
                    ctx0=blend('ctxst0', self.legacy), ctx1=blend('ctxst1', self.legacy))

