"""Multi-parametric encoder with block-level MSASPP (paper Sec. 3.2).

The encoder produces a five-level pyramid {F1, ..., F5} from a MedNeXt
backbone. At the bottleneck F5 a Multi-Scale Atrous Spatial Pyramid Pooling
module applies dilated convolutions at rates {1, 2, 4, 6} plus a global
average pooling branch (Eq. 3), fuses them with a Res2Net-style hierarchical
split-and-fuse operation and projects back with a 1x1(x1) convolution (Eq. 4).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import (ConvNormAct, MedNeXtBlock, MedNeXtDown, adaptive_avg_pool_nd,
                     conv_nd, resize_to)


class Res2Fuse(nn.Module):
    """Res2Net hierarchical split-and-fuse: y1 = x1, y_i = K_i(x_i + y_{i-1})."""

    def __init__(self, spatial_dims, channels, scales):
        super().__init__()
        assert channels % scales == 0, "channels must be divisible by scales"
        self.scales = scales
        width = channels // scales
        self.convs = nn.ModuleList(
            [ConvNormAct(spatial_dims, width, width, 3) for _ in range(scales - 1)])

    def forward(self, x):
        xs = torch.chunk(x, self.scales, dim=1)
        ys = [xs[0]]
        prev = None
        for i in range(1, self.scales):
            inp = xs[i] if prev is None else xs[i] + prev
            prev = self.convs[i - 1](inp)
            ys.append(prev)
        return torch.cat(ys, dim=1)


class MSASPP(nn.Module):
    """Block-level Multi-Scale Atrous Spatial Pyramid Pooling (Eqs. 3-4)."""

    def __init__(self, spatial_dims, channels, rates: Sequence[int] = (1, 2, 4, 6),
                 branch_channels: int | None = None):
        super().__init__()
        n_branches = len(rates) + 1  # dilated branches + GAP branch
        bc = branch_channels or max(channels // n_branches, 4)
        self.branches = nn.ModuleList(
            [ConvNormAct(spatial_dims, channels, bc, 3, dilation=r) for r in rates])
        self.gap = nn.Sequential(adaptive_avg_pool_nd(spatial_dims)(1),
                                 conv_nd(spatial_dims)(channels, bc, 1),
                                 nn.GELU())
        self.res2 = Res2Fuse(spatial_dims, bc * n_branches, n_branches)
        self.project = ConvNormAct(spatial_dims, bc * n_branches, channels, 1)

    def forward(self, f5):
        size = f5.shape[2:]
        outs = [b(f5) for b in self.branches]
        outs.append(resize_to(self.gap(f5), size, mode="nearest"))
        return self.project(self.res2(torch.cat(outs, dim=1)))


class MQDAEncoder(nn.Module):
    """MedNeXt encoder + MSASPP bottleneck, returning [F1, ..., F5']."""

    def __init__(self, spatial_dims=3, in_channels=4, base_channels=32,
                 block_counts: Sequence[int] = (3, 4, 8, 8, 8),
                 exp_ratios: Sequence[int] = (3, 4, 8, 8, 8),
                 kernel_size=3, aspp_rates: Sequence[int] = (1, 2, 4, 6)):
        super().__init__()
        assert len(block_counts) == 5 and len(exp_ratios) == 5
        self.spatial_dims = spatial_dims
        chs = [base_channels * 2 ** i for i in range(5)]
        self.channels = chs
        self.stem = conv_nd(spatial_dims)(in_channels, chs[0], 1)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(5):
            self.stages.append(nn.Sequential(*[
                MedNeXtBlock(spatial_dims, chs[i], exp_ratios[i], kernel_size)
                for _ in range(block_counts[i])]))
            if i < 4:
                self.downs.append(MedNeXtDown(spatial_dims, chs[i], chs[i + 1],
                                              exp_ratios[i + 1], kernel_size))
        self.msaspp = MSASPP(spatial_dims, chs[4], aspp_rates)

    @property
    def bottleneck_channels(self) -> int:
        return self.channels[-1]

    def forward(self, x):
        feats = []
        h = self.stem(x)
        for i in range(5):
            h = self.stages[i](h)
            if i < 4:
                feats.append(h)
                h = self.downs[i](h)
        feats.append(self.msaspp(h))  # F5'
        return feats
