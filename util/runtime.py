import random
import numpy as np
import torch
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed=torch.initial_seed() % 2**32
    random.seed(seed);np.random.seed(seed)


def resolve_device(value='auto'):
    value='cuda' if value=='auto' and torch.cuda.is_available() else 'cpu' if value=='auto' else value
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for small smoke tests')
    return torch.device(value)


def pad_inputs(im0,im1,event,multiple):
    h,w=im0.shape[-2:]
    padding=(0,(-w)%multiple,0,(-h)%multiple)
    return tuple(F.pad(x,padding) for x in (im0,im1,event)),(h,w)

