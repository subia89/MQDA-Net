"""Deterministic 2D-to-3D input adaptation A_2D->3D (paper Sec. 3.7).

The classification cohort is two-dimensional, but it must reach the same 3D
encoder as the segmentation volumes. The adaptation has no trainable
parameters and works in three steps on a batch (N, 1, 224, 224):

1. resize in plane to the encoder resolution      -> (N, 1, 128, 128)
2. replicate along the depth axis                 -> (N, 1, 128, 128, 16)
3. replicate across the four sequence channels    -> (N, 4, 128, 128, 16)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def adapt_2d_to_3d(x: torch.Tensor, size: int = 128, depth: int = 16,
                   channels: int = 4) -> torch.Tensor:
    """(N, C, H, W) -> (N, ``channels``, size, size, depth)."""
    if x.dim() != 4:
        raise ValueError(f"expected a 2D batch (N, C, H, W), got shape {tuple(x.shape)}")
    y = F.interpolate(x.float(), size=(size, size), mode="bilinear", align_corners=False)
    y = y.unsqueeze(-1).expand(-1, -1, -1, -1, depth)
    if y.shape[1] == 1 and channels > 1:
        y = y.expand(-1, channels, -1, -1, -1)
    elif y.shape[1] != channels:
        raise ValueError(f"cannot adapt {y.shape[1]} channels to {channels}")
    return y.contiguous()
