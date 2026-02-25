"""Re-implementation of the Histoformer restoration network for PyTorch 2.9."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Sequence, Tuple
from torch.utils.checkpoint import checkpoint_sequential

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _pad_factor(x: torch.Tensor, factor: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    hw = x.shape[-1]
    if hw % factor == 0:
        return x, (0, 0)
    target = (hw // factor + 1) * factor
    pad = target - hw
    return F.pad(x, (0, pad), value=0.0), (0, pad)


def _unpad_factor(x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
    if pad[1] == 0:
        return x
    return x[..., pad[0] : x.shape[-1] - pad[1]]


class LayerNorm2d(nn.Module):
    """Applies LayerNorm over the channel dimension of a 4-D tensor."""

    def __init__(self, channels: int, bias_free: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.bias_free = bias_free
        self.scale = nn.Parameter(torch.ones(channels))
        if not bias_free:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias_free:
            return normed * self.scale.view(1, -1, 1, 1)
        return normed * self.scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DualScaleGatedFFN(nn.Module):
    """Dual-scale gated feed-forward module inspired by the original implementation."""

    def __init__(self, channels: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden = int(round(channels * expansion))
        hidden = max(hidden, channels)
        hidden = ((hidden + 3) // 4) * 4
        self.proj_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.shuffle = nn.PixelShuffle(2)
        mid_channels = hidden * 2 // (2 * 2)
        self.dwconv_primary = nn.Conv2d(
            mid_channels // 2,
            mid_channels // 2,
            kernel_size=5,
            padding=2,
            groups=mid_channels // 2,
            bias=bias,
        )
        self.dwconv_context = nn.Conv2d(
            mid_channels // 2,
            mid_channels // 2,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=mid_channels // 2,
            bias=bias,
        )
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.shuffle(x)
        x_primary, x_context = torch.chunk(x, 2, dim=1)
        x_primary = self.dwconv_primary(x_primary)
        x_context = F.mish(self.dwconv_context(x_context))
        x = x_primary * x_context
        x = self.unshuffle(x)
        x = self.proj_out(x)
        return x


class HistogramSelfAttention(nn.Module):
    """Dynamic-range histogram self-attention (DHSA)."""

    def __init__(self, channels: int, heads: int, bias: bool) -> None:
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 5, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(
            channels * 5,
            channels * 5,
            kernel_size=3,
            padding=1,
            groups=channels * 5,
            bias=bias,
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def _reshape_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, factor_first: bool
    ) -> torch.Tensor:
        b, c, hw = q.shape
        q, pad = _pad_factor(q, self.heads)
        k, _ = _pad_factor(k, self.heads)
        v, _ = _pad_factor(v, self.heads)
        grid = q.shape[-1] // self.heads
        if factor_first:
            pattern = "b (h c) (f g) -> b h (c f) g"
        else:
            pattern = "b (h c) (g f) -> b h (c f) g"
        q = rearrange(q, pattern, h=self.heads, g=grid)
        k = rearrange(k, pattern, h=self.heads, g=grid)
        v = rearrange(v, pattern, h=self.heads, g=grid)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (c f) g -> b (h c) (f g)", f=self.heads, g=grid)
        out = _unpad_factor(out, pad)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        half = c // 2
        histogram_slice = x[:, :half].contiguous()
        sorted_h, idx_h = histogram_slice.sort(dim=-2)
        sorted_hw, idx_w = sorted_h.sort(dim=-1)
        x_sorted = x.clone()
        x_sorted[:, :half] = sorted_hw
        qkv = self.depthwise(self.qkv(x_sorted))
        q1, k1, q2, k2, v = torch.chunk(qkv, 5, dim=1)
        v_flat, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)
        out_primary = self._reshape_attention(q1, k1, v_flat, factor_first=True)
        out_context = self._reshape_attention(q2, k2, v_flat, factor_first=False)
        out_primary = torch.scatter(out_primary, 2, idx, out_primary)
        out_context = torch.scatter(out_context, 2, idx, out_context)
        out = out_primary.view(b, c, h, w) * out_context.view(b, c, h, w)
        out = self.proj_out(out)
        recovered = out[:, :half]
        recovered = torch.scatter(recovered, -1, idx_w, recovered)
        recovered = torch.scatter(recovered, -2, idx_h, recovered)
        out[:, :half] = recovered
        return out


class TransformerStage(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        expansion: float,
        bias: bool,
        layernorm_type: str,
    ) -> None:
        super().__init__()
        bias_free = layernorm_type.lower() == "biasfree"
        self.norm_attn = LayerNorm2d(channels, bias_free=bias_free)
        self.attn = HistogramSelfAttention(channels, heads, bias)
        self.norm_ffn = LayerNorm2d(channels, bias_free=bias_free)
        self.mlp = DualScaleGatedFFN(channels, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_ffn(x))
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)


class SkipEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


@dataclass
class RestorationConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_dim: int = 48
    num_blocks: Sequence[int] = (4, 6, 6, 8)
    num_heads: Sequence[int] = (1, 2, 4, 8)
    num_refine: int = 4
    ffn_expansion: float = 2.66
    bias: bool = False
    layernorm_type: str = "with_bias"


class HistoformerRestoration(nn.Module):
    def __init__(self, cfg: RestorationConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or RestorationConfig()
        dim = cfg.base_dim
        self.patch_embed = nn.Conv2d(cfg.in_channels, dim, kernel_size=3, padding=1, bias=cfg.bias)
        self.encoder1 = nn.Sequential(
            *[
                TransformerStage(dim, cfg.num_heads[0], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[0])
            ]
        )
        self.down12 = Downsample(dim)
        self.skip1 = SkipEmbedding(cfg.in_channels, cfg.in_channels, bias=cfg.bias)
        self.reduce12 = nn.Conv2d(dim * 2 + cfg.in_channels, dim * 2, kernel_size=1, bias=cfg.bias)

        dim2 = dim * 2
        self.encoder2 = nn.Sequential(
            *[
                TransformerStage(dim2, cfg.num_heads[1], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[1])
            ]
        )
        self.down23 = Downsample(dim2)
        self.skip2 = SkipEmbedding(cfg.in_channels, cfg.in_channels, bias=cfg.bias)
        self.reduce23 = nn.Conv2d(dim * 4 + cfg.in_channels, dim * 4, kernel_size=1, bias=cfg.bias)

        dim3 = dim * 4
        self.encoder3 = nn.Sequential(
            *[
                TransformerStage(dim3, cfg.num_heads[2], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[2])
            ]
        )
        self.down34 = Downsample(dim3)
        self.skip3 = SkipEmbedding(cfg.in_channels, cfg.in_channels, bias=cfg.bias)
        self.reduce34 = nn.Conv2d(dim * 8 + cfg.in_channels, dim * 8, kernel_size=1, bias=cfg.bias)

        dim4 = dim * 8
        self.latent = nn.Sequential(
            *[
                TransformerStage(dim4, cfg.num_heads[3], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[3])
            ]
        )

        self.up43 = Upsample(dim4)
        self.reduce_dec3 = nn.Conv2d(dim4, dim3, kernel_size=1, bias=cfg.bias)
        self.decoder3 = nn.Sequential(
            *[
                TransformerStage(dim3, cfg.num_heads[2], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[2])
            ]
        )

        self.up32 = Upsample(dim3)
        self.reduce_dec2 = nn.Conv2d(dim3, dim2, kernel_size=1, bias=cfg.bias)
        self.decoder2 = nn.Sequential(
            *[
                TransformerStage(dim2, cfg.num_heads[1], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[1])
            ]
        )

        self.up21 = Upsample(dim2)
        self.decoder1 = nn.Sequential(
            *[
                TransformerStage(dim2, cfg.num_heads[0], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_blocks[0])
            ]
        )
        self.refine = nn.Sequential(
            *[
                TransformerStage(dim2, cfg.num_heads[0], cfg.ffn_expansion, cfg.bias, cfg.layernorm_type)
                for _ in range(cfg.num_refine)
            ]
        )
        self.output = nn.Conv2d(dim2, cfg.out_channels, kernel_size=3, padding=1, bias=cfg.bias)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x1 = self.patch_embed(x)
        x1 = self.encoder1(x1)

        x2 = self.down12(x1)
        skip1 = self.skip1(inp)
        x2 = self.reduce12(torch.cat([x2, skip1], dim=1))
        x2 = self.encoder2(x2)

        x3 = self.down23(x2)
        skip2 = self.skip2(skip1)
        x3 = self.reduce23(torch.cat([x3, skip2], dim=1))
        x3 = self.encoder3(x3)

        x4 = self.down34(x3)
        skip3 = self.skip3(skip2)
        x4 = self.reduce34(torch.cat([x4, skip3], dim=1))
        x4 = self.latent(x4)

        d3 = self.up43(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.reduce_dec3(d3)
        d3 = self.decoder3(d3)

        d2 = self.up32(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.reduce_dec2(d2)
        d2 = self.decoder2(d2)

        d1 = self.up21(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.decoder1(d1)
        d1 = self.refine(d1)
        out = self.output(d1)
        return out + inp


def build_restoration_model(cfg_dict: dict | None = None) -> HistoformerRestoration:
    # Filter out non-RestorationConfig keys (e.g., gradient_checkpointing)
    cfg = None
    if cfg_dict:
        rc_field_names = {f.name for f in fields(RestorationConfig)}
        rc_kwargs = {k: v for k, v in cfg_dict.items() if k in rc_field_names}
        cfg = RestorationConfig(**rc_kwargs)
    m = HistoformerRestoration(cfg)
    # Optional gradient checkpointing to reduce VRAM peak usage
    gc = False
    seg = 1
    if cfg_dict:
        gc = bool(cfg_dict.get("gradient_checkpointing", False))
        seg = int(cfg_dict.get("checkpoint_segments", 1))
    if gc:
        # Wrap large sequential stacks with checkpoint_sequential
        def wrap_seq(module: nn.Sequential) -> nn.Module:
            layers = list(module.children())
            if not layers:
                return module
            class _CheckpointSeq(nn.Module):
                def __init__(self, layers: list[nn.Module], segments: int) -> None:
                    super().__init__()
                    self.seq = nn.Sequential(*layers)
                    self.layers = layers
                    self.segments = max(1, segments)
                def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                    # Ensure segments does not exceed number of layers to avoid zero step
                    segs = min(self.segments, max(1, len(self.layers)))
                    try:
                        return checkpoint_sequential(self.layers, segs, x, use_reentrant=False)
                    except TypeError:
                        # Fallback for older signature without use_reentrant
                        return checkpoint_sequential(self.layers, segs, x)
            return _CheckpointSeq(layers, seg)

        m.encoder1 = wrap_seq(m.encoder1)  # type: ignore[assignment]
        m.encoder2 = wrap_seq(m.encoder2)  # type: ignore[assignment]
        m.encoder3 = wrap_seq(m.encoder3)  # type: ignore[assignment]
        m.latent = wrap_seq(m.latent)      # type: ignore[assignment]
        m.decoder3 = wrap_seq(m.decoder3)  # type: ignore[assignment]
        m.decoder2 = wrap_seq(m.decoder2)  # type: ignore[assignment]
        m.decoder1 = wrap_seq(m.decoder1)  # type: ignore[assignment]
        m.refine = wrap_seq(m.refine)      # type: ignore[assignment]
    return m
