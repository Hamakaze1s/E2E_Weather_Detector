"""
Histoformer-inspired Restoration Network
==========================================

A U-Net style transformer with Dynamic-Range Histogram Self-Attention (DHSA)
and Dual-Scale Gated FFN, re-implemented in pure PyTorch.

Reference architecture (medium_32g_251225 config):
    base_dim    = 16
    num_blocks  = [1, 2, 2, 2]
    num_heads   = [1, 1, 2, 2]
    num_refine  = 1
    ffn_expansion = 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint_sequential


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to(x: torch.Tensor, factor: int) -> Tuple[torch.Tensor, int]:
    hw = x.shape[-1]
    if hw % factor == 0:
        return x, 0
    pad = (hw // factor + 1) * factor - hw
    return F.pad(x, (0, pad)), pad


def _unpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    return x if pad == 0 else x[..., :-pad]


class LayerNorm2d(nn.Module):
    def __init__(self, ch: int, bias_free: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps, self.bias_free = eps, bias_free
        self.scale = nn.Parameter(torch.ones(ch))
        self.bias  = None if bias_free else nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m  = x.mean(1, keepdim=True)
        v  = (x - m).pow(2).mean(1, keepdim=True)
        xn = (x - m) / (v + self.eps).sqrt()
        out = xn * self.scale.view(1, -1, 1, 1)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Dual-Scale Gated FFN
# ---------------------------------------------------------------------------

class DualScaleGatedFFN(nn.Module):
    def __init__(self, ch: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden = max(((int(round(ch * expansion)) + 3) // 4) * 4, ch)
        self.proj_in  = nn.Conv2d(ch, hidden * 2, 1, bias=bias)
        self.shuffle   = nn.PixelShuffle(2)
        mc = hidden * 2 // 4
        self.dw_p = nn.Conv2d(mc // 2, mc // 2, 5, padding=2,            groups=mc // 2, bias=bias)
        self.dw_c = nn.Conv2d(mc // 2, mc // 2, 3, padding=2, dilation=2, groups=mc // 2, bias=bias)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj_out  = nn.Conv2d(hidden, ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.shuffle(self.proj_in(x))
        xp, xc = torch.chunk(x, 2, dim=1)
        x  = self.dw_p(xp) * F.mish(self.dw_c(xc))
        return self.proj_out(self.unshuffle(x))


# ---------------------------------------------------------------------------
# Histogram Self-Attention (DHSA)
# ---------------------------------------------------------------------------

class HistogramSelfAttention(nn.Module):
    """Dynamic-range histogram self-attention."""

    def __init__(self, ch: int, heads: int, bias: bool) -> None:
        super().__init__()
        self.heads = heads
        self.temp  = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv   = nn.Conv2d(ch, ch * 5, 1, bias=bias)
        self.dw    = nn.Conv2d(ch * 5, ch * 5, 3, padding=1, groups=ch * 5, bias=bias)
        self.proj  = nn.Conv2d(ch, ch, 1, bias=bias)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, first: bool) -> torch.Tensor:
        b, c, n = q.shape
        q, pad = _pad_to(q, self.heads)
        k, _   = _pad_to(k, self.heads)
        v, _   = _pad_to(v, self.heads)
        g = q.shape[-1] // self.heads
        pat = "b (h c) (f g) -> b h (c f) g" if first else "b (h c) (g f) -> b h (c f) g"
        q = F.normalize(rearrange(q, pat, h=self.heads, g=g), dim=-1)
        k = F.normalize(rearrange(k, pat, h=self.heads, g=g), dim=-1)
        v = rearrange(v, pat, h=self.heads, g=g)
        out = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temp, -1), v)
        out = rearrange(out, "b h (c f) g -> b (h c) (f g)", f=self.heads, g=g)
        return _unpad(out, pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        half = c // 2
        hs    = x[:, :half]
        sh, ih = hs.sort(-2);  shw, iw = sh.sort(-1)
        xs = x.clone(); xs[:, :half] = shw
        qkv  = self.dw(self.qkv(xs))
        q1, k1, q2, k2, v = qkv.chunk(5, 1)
        vf, idx = v.view(b, c, -1).sort(-1)
        q1 = q1.view(b, c, -1).gather(2, idx)
        k1 = k1.view(b, c, -1).gather(2, idx)
        q2 = q2.view(b, c, -1).gather(2, idx)
        k2 = k2.view(b, c, -1).gather(2, idx)
        o1 = self._attn(q1, k1, vf, True).scatter(2, idx, self._attn(q1, k1, vf, True))
        o2 = self._attn(q2, k2, vf, False).scatter(2, idx, self._attn(q2, k2, vf, False))
        out = (o1 * o2).view(b, c, h, w)
        out = self.proj(out)
        # undo histogram sort
        rec = out[:, :half]
        rec = rec.scatter(-1, iw, rec).scatter(-2, ih, rec)
        out[:, :half] = rec
        return out


# ---------------------------------------------------------------------------
# Transformer Stage
# ---------------------------------------------------------------------------

class TransformerStage(nn.Module):
    def __init__(self, ch: int, heads: int, exp: float, bias: bool, ln: str) -> None:
        super().__init__()
        bf = ln.lower() == "biasfree"
        self.n1   = LayerNorm2d(ch, bf)
        self.attn = HistogramSelfAttention(ch, heads, bias)
        self.n2   = LayerNorm2d(ch, bf)
        self.ffn  = DualScaleGatedFFN(ch, exp, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SkipEmbed(nn.Module):
    def __init__(self, ic: int, oc: int, bias: bool) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(ic, oc, 1, bias=bias),
            nn.Conv2d(oc, oc, 3, padding=1, groups=oc, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ---------------------------------------------------------------------------
# Full restoration U-Net
# ---------------------------------------------------------------------------

@dataclass
class RestorationConfig:
    in_channels:   int             = 3
    out_channels:  int             = 3
    base_dim:      int             = 48
    num_blocks:    Sequence[int]   = (4, 6, 6, 8)
    num_heads:     Sequence[int]   = (1, 2, 4, 8)
    num_refine:    int             = 4
    ffn_expansion: float           = 2.66
    bias:          bool            = False
    layernorm_type: str            = "with_bias"


class HistoformerRestoration(nn.Module):
    """
    U-Net restoration model with histogram self-attention.
    Residual learning: output = network(x) + x.
    """

    def __init__(self, cfg: RestorationConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or RestorationConfig()
        d, ic, oc = cfg.base_dim, cfg.in_channels, cfg.out_channels
        b, h, exp, bias, ln = cfg.num_blocks, cfg.num_heads, cfg.ffn_expansion, cfg.bias, cfg.layernorm_type
        NR = cfg.num_refine

        def stage(ch, ni, nhead): return nn.Sequential(
            *[TransformerStage(ch, nhead, exp, bias, ln) for _ in range(ni)])

        self.patch    = nn.Conv2d(ic, d, 3, padding=1, bias=bias)
        self.enc1     = stage(d,    b[0], h[0])
        self.dn12     = Downsample(d)
        self.sk1      = SkipEmbed(ic, ic, bias)
        self.red12    = nn.Conv2d(d*2+ic, d*2, 1, bias=bias)
        self.enc2     = stage(d*2,  b[1], h[1])
        self.dn23     = Downsample(d*2)
        self.sk2      = SkipEmbed(ic, ic, bias)
        self.red23    = nn.Conv2d(d*4+ic, d*4, 1, bias=bias)
        self.enc3     = stage(d*4,  b[2], h[2])
        self.dn34     = Downsample(d*4)
        self.sk3      = SkipEmbed(ic, ic, bias)
        self.red34    = nn.Conv2d(d*8+ic, d*8, 1, bias=bias)
        self.latent   = stage(d*8,  b[3], h[3])
        self.up43     = Upsample(d*8)
        self.rdc3     = nn.Conv2d(d*8, d*4, 1, bias=bias)
        self.dec3     = stage(d*4,  b[2], h[2])
        self.up32     = Upsample(d*4)
        self.rdc2     = nn.Conv2d(d*4, d*2, 1, bias=bias)
        self.dec2     = stage(d*2,  b[1], h[1])
        self.up21     = Upsample(d*2)
        self.dec1     = stage(d*2,  b[0], h[0])
        self.refine   = stage(d*2,  NR,   h[0])
        self.out      = nn.Conv2d(d*2, oc, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x1  = self.enc1(self.patch(x))
        s1  = self.sk1(inp)
        x2  = self.enc2(self.red12(torch.cat([self.dn12(x1), s1], 1)))
        s2  = self.sk2(s1)
        x3  = self.enc3(self.red23(torch.cat([self.dn23(x2), s2], 1)))
        s3  = self.sk3(s2)
        x4  = self.latent(self.red34(torch.cat([self.dn34(x3), s3], 1)))
        d3  = self.dec3(self.rdc3(torch.cat([self.up43(x4), x3], 1)))
        d2  = self.dec2(self.rdc2(torch.cat([self.up32(d3), x2], 1)))
        d1  = self.refine(self.dec1(torch.cat([self.up21(d2), x1], 1)))
        return self.out(d1) + inp   # residual connection


def build_restoration_model(cfg_dict: dict | None = None) -> HistoformerRestoration:
    cfg = None
    if cfg_dict:
        valid = {f.name for f in fields(RestorationConfig)}
        cfg   = RestorationConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    return HistoformerRestoration(cfg)
