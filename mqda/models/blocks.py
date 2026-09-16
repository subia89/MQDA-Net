"""Dimension-agnostic building blocks.

Every module in MQDA-Net takes a ``spatial_dims`` argument (2 or 3) so the
same code serves the volumetric BraTS setting (4 x 128^3) and the
two-dimensional slice datasets (FigShare, Br35H, 3K-DS, 7K-DS at 224^2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(spatial_dims: int):
    return {2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]


def conv_transpose_nd(spatial_dims: int):
    return {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[spatial_dims]


def adaptive_avg_pool_nd(spatial_dims: int):
    return {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[spatial_dims]


def interp_mode(spatial_dims: int) -> str:
    return {2: "bilinear", 3: "trilinear"}[spatial_dims]


def resize_to(x: torch.Tensor, size, mode: str | None = None) -> torch.Tensor:
    """Resize a (N, C, *spatial) tensor to ``size`` (nearest for masks if asked)."""
    sd = x.dim() - 2
    if tuple(x.shape[2:]) == tuple(size):
        return x
    mode = mode or interp_mode(sd)
    kwargs = {} if mode in ("nearest", "area") else {"align_corners": False}
    return F.interpolate(x, size=size, mode=mode, **kwargs)


class ConvNormAct(nn.Module):
    def __init__(self, spatial_dims, in_ch, out_ch, kernel_size=3, stride=1,
                 dilation=1, groups=1, norm_groups=8, act=True):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = conv_nd(spatial_dims)(in_ch, out_ch, kernel_size, stride,
                                          pad, dilation=dilation, groups=groups,
                                          bias=False)
        self.norm = nn.GroupNorm(math.gcd(norm_groups, out_ch), out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MedNeXtBlock(nn.Module):
    """MedNeXt block (Roy et al., MICCAI 2023).

    depthwise k^d conv -> GroupNorm -> 1x1 expand (x R) -> GELU -> 1x1 compress,
    with a residual connection.
    """

    def __init__(self, spatial_dims, channels, exp_r=4, kernel_size=3):
        super().__init__()
        conv = conv_nd(spatial_dims)
        self.dw = conv(channels, channels, kernel_size, padding=kernel_size // 2,
                       groups=channels)
        self.norm = nn.GroupNorm(channels, channels)
        self.expand = conv(channels, channels * exp_r, 1)
        self.act = nn.GELU()
        self.compress = conv(channels * exp_r, channels, 1)

    def forward(self, x):
        return x + self.compress(self.act(self.expand(self.norm(self.dw(x)))))


class MedNeXtDown(nn.Module):
    """Strided MedNeXt block that halves resolution and doubles channels."""

    def __init__(self, spatial_dims, in_ch, out_ch, exp_r=4, kernel_size=3):
        super().__init__()
        conv = conv_nd(spatial_dims)
        self.dw = conv(in_ch, in_ch, kernel_size, stride=2,
                       padding=kernel_size // 2, groups=in_ch)
        self.norm = nn.GroupNorm(in_ch, in_ch)
        self.expand = conv(in_ch, in_ch * exp_r, 1)
        self.act = nn.GELU()
        self.compress = conv(in_ch * exp_r, out_ch, 1)
        self.skip = conv(in_ch, out_ch, 1, stride=2)

    def forward(self, x):
        y = self.compress(self.act(self.expand(self.norm(self.dw(x)))))
        return y + self.skip(x)


class MedNeXtUp(nn.Module):
    """Transposed MedNeXt block that doubles resolution and halves channels."""

    def __init__(self, spatial_dims, in_ch, out_ch, exp_r=4, kernel_size=3):
        super().__init__()
        tconv = conv_transpose_nd(spatial_dims)
        conv = conv_nd(spatial_dims)
        self.dw = tconv(in_ch, in_ch, kernel_size, stride=2,
                        padding=kernel_size // 2, output_padding=1, groups=in_ch)
        self.norm = nn.GroupNorm(in_ch, in_ch)
        self.expand = conv(in_ch, in_ch * exp_r, 1)
        self.act = nn.GELU()
        self.compress = conv(in_ch * exp_r, out_ch, 1)
        self.skip = tconv(in_ch, out_ch, 1, stride=2, output_padding=1)

    def forward(self, x):
        y = self.compress(self.act(self.expand(self.norm(self.dw(x)))))
        return y + self.skip(x)
