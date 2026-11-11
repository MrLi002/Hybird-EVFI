"""DSER++-inspired patch retrieval with separate query/source masks and full gating."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def make_basr_masks(w):
    size = w['fuse'].shape[-2:]
    up = lambda m: F.interpolate(m, size=size, mode='nearest')
    bad0 = torch.maximum(up(w['a']['hole']), w['a']['diff'])
    bad1 = torch.maximum(up(w['b']['hole']), w['b']['diff'])
    bad0 = torch.maximum(bad0, 1-w['a']['in_bounds']).clamp(0, 1).detach()
    bad1 = torch.maximum(bad1, 1-w['b']['in_bounds']).clamp(0, 1).detach()
    repair = torch.maximum(bad0, bad1)
    support0, support1 = bad1*(1-bad0), bad0*(1-bad1)
    available = ((support0+support1) > 0).to(repair.dtype)
    return dict(bad0=bad0, bad1=bad1, repair=repair, support0=support0,
                support1=support1, repair_gate=repair*available)


def safe_key_attention(logits, values, key_valid):
    if logits.shape[-1] == 0:
        raise ValueError('at least one placeholder key token is required')
    has_key = key_valid.any(-1)
    safe_valid = key_valid.clone()
    safe_valid[~has_key, 0] = True
    logits = logits.masked_fill(~safe_valid[:, None, None, :], float('-inf'))
    weights = logits.float().softmax(-1).to(values.dtype)
    result = weights @ values
    return result * has_key[:, None, None, None].to(result.dtype), has_key


def valid_support_keys(support, feature_size, patch_size, support_ratio=1.0):
    small = F.adaptive_avg_pool2d(support.float(), feature_size)
    reliable = (small >= 1-1e-6).float()
    ratio = F.avg_pool2d(reliable, patch_size, patch_size).flatten(1)
    return ratio >= support_ratio-1e-6


def effective_repair_gate(masks, support_ratio=1.0):
    """Union of the two actual BASR output gates, including empty-key bypasses."""
    height,width=masks['repair_gate'].shape[-2:]
    effective=torch.zeros_like(masks['repair_gate'])
    for down,patch in ((4,8),(2,16)):
        size=(height//down,width//down)
        has0=valid_support_keys(masks['support0'],size,patch,support_ratio).any(-1)
        has1=valid_support_keys(masks['support1'],size,patch,support_ratio).any(-1)
        output_size=(2*size[0],2*size[1])
        gate=F.interpolate(masks['repair_gate'],size=output_size,mode='nearest')
        gate=gate*(has0|has1)[:,None,None,None].to(gate.dtype)
        effective=torch.maximum(effective,F.interpolate(gate,size=(height,width),mode='nearest'))
    return effective


def sine_position(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000)/dim))
    result = torch.zeros(length, dim, device=device)
    result[:, 0::2] = torch.sin(position*frequency)
    result[:, 1::2] = torch.cos(position*frequency[:dim//2])
    return result[None].to(dtype)


class TransformerBlock(nn.Module):
    def __init__(self, patch_size, dim, channels, num_heads=4, chunk_size=256, support_ratio=1.0):
        super().__init__()
        assert num_heads > 0 and dim % num_heads == 0
        self.patch_size, self.dim, self.heads = patch_size, dim, num_heads
        self.chunk_size, self.support_ratio = chunk_size, support_ratio
        patch_dim = channels*patch_size**2
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.img0_to_kv = nn.Linear(dim, 2*dim)
        self.img1_to_q = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, patch_dim))

    def forward(self, source, query, support):
        batch, channels, height, width = query.shape
        p = self.patch_size
        if height % p or width % p or source.shape != query.shape:
            raise ValueError('BASR expects equal feature shapes divisible by patch size')
        source_tokens = rearrange(source, 'b c (h p) (w q) -> b (h w) (p q c)', p=p, q=p)
        query_tokens = rearrange(query, 'b c (h p) (w q) -> b (h w) (p q c)', p=p, q=p)
        source_tokens = self.to_patch_embedding(source_tokens)
        query_tokens = self.to_patch_embedding(query_tokens)
        position = sine_position(source_tokens.shape[1], self.dim, source.device, source_tokens.dtype)
        source_tokens, query_tokens = source_tokens+position, query_tokens+position
        kv = self.img0_to_kv(source_tokens).reshape(batch, -1, 2, self.heads, self.dim//self.heads)
        keys, values = kv.permute(2, 0, 3, 1, 4).unbind(0)
        queries = self.img1_to_q(query_tokens).reshape(batch, -1, self.heads, self.dim//self.heads).transpose(1, 2)
        # Downsample conservatively before patch selection: only reliable support contributes keys.
        key_valid = valid_support_keys(support, (height,width), p, self.support_ratio)
        outputs = []
        for begin in range(0, queries.shape[2], self.chunk_size):
            q = queries[:, :, begin:begin+self.chunk_size]
            logits = (q @ keys.transpose(-2, -1)) / math.sqrt(self.dim//self.heads)
            values_out, has_key = safe_key_attention(logits, values, key_valid)
            outputs.append(values_out)
        tokens = torch.cat(outputs, 2).transpose(1, 2).reshape(batch, -1, self.dim)
        tokens = self.mlp(tokens) * has_key[:, None, None].to(tokens.dtype)
        result = rearrange(tokens, 'b (h w) (p q c) -> b c (h p) (w q)',
                           h=height//p, w=width//p, p=p, q=p, c=channels)
        return result, has_key


class DetailsSup(nn.Module):
    def __init__(self, patch_size, dim, channels, heads=4, chunk_size=256, support_ratio=1.0):
        super().__init__()
        self.transformer_block = TransformerBlock(patch_size, dim, channels, heads, chunk_size, support_ratio)
        self.upsample = nn.ConvTranspose2d(2*channels, channels, 2, 2)
        self.fusion_block = nn.Sequential(nn.Conv2d(5*channels, channels, 3, padding=1),
                                          nn.LeakyReLU(.1), nn.Conv2d(channels, 4*channels, 3, padding=1))
        nn.init.zeros_(self.fusion_block[-1].weight)
        nn.init.zeros_(self.fusion_block[-1].bias)

    def forward(self, imgt, ref0, ref1, f_up, masks):
        a, has0 = self.transformer_block(ref0, imgt, masks['support0'])
        b, has1 = self.transformer_block(ref1, imgt, masks['support1'])
        up = self.upsample(torch.cat((imgt+a, imgt+b), 1))
        delta = self.fusion_block(torch.cat((up, f_up), 1))
        gate = F.interpolate(masks['repair_gate'], size=f_up.shape[-2:], mode='nearest')
        gate = gate * (has0|has1)[:, None, None, None].to(gate.dtype)
        return f_up + gate*delta

