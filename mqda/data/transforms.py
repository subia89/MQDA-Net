"""Augmentation used in the paper (Sec. 3.7), implemented for 2D and 3D tensors.

random affine (rotation up to +/-15 deg, scale 0.85-1.15), elastic deformation,
intensity shift and Gaussian noise (sigma = 0.05). Image and label are
transformed jointly (bilinear / nearest).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def _rotation_matrix(sd, max_deg):
    if sd == 2:
        a = math.radians(random.uniform(-max_deg, max_deg))
        return torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    r = torch.eye(3)
    for axes in ((0, 1), (0, 2), (1, 2)):
        a = math.radians(random.uniform(-max_deg, max_deg))
        m = torch.eye(3)
        i, j = axes
        m[i, i], m[i, j], m[j, i], m[j, j] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
        r = m @ r
    return r


class Augment:
    def __init__(self, p_affine=0.5, max_rotation=15.0, scale=(0.85, 1.15), p_elastic=0.2,
                 elastic_alpha=0.03, elastic_grid=6, p_shift=0.5, shift=0.1,
                 p_noise=0.5, noise_std=0.05, p_flip=0.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def __call__(self, image: torch.Tensor, label: torch.Tensor | None):
        sd = image.dim() - 1
        x = image.unsqueeze(0)
        y = label[None, None].float() if label is not None else None
        grid = None
        if random.random() < self.p_affine:
            rot = _rotation_matrix(sd, self.max_rotation)
            s = random.uniform(*self.scale)
            theta = torch.cat([rot / s, torch.zeros(sd, 1)], 1).unsqueeze(0)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
        if random.random() < self.p_elastic:
            if grid is None:
                eye = torch.cat([torch.eye(sd), torch.zeros(sd, 1)], 1).unsqueeze(0)
                grid = F.affine_grid(eye, x.shape, align_corners=False)
            coarse = torch.randn(1, sd, *([self.elastic_grid] * sd)) * self.elastic_alpha
            mode = "trilinear" if sd == 3 else "bilinear"
            disp = F.interpolate(coarse, size=x.shape[2:], mode=mode, align_corners=False)
            grid = grid + disp.movedim(1, -1)
        if grid is not None:
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            if y is not None:
                y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        if self.p_flip > 0:
            for d in range(sd):
                if random.random() < self.p_flip:
                    x = x.flip(2 + d)
                    if y is not None:
                        y = y.flip(2 + d)
        x = x[0]
        if random.random() < self.p_shift:
            x = x + torch.empty(x.shape[0], *[1] * sd).uniform_(-self.shift, self.shift)
        if random.random() < self.p_noise:
            x = x + torch.randn_like(x) * self.noise_std
        return x, (y[0, 0].long() if y is not None else None)


def zscore(img: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Per-channel z-score over non-zero (brain) voxels."""
    out = torch.zeros_like(img, dtype=torch.float32)
    for c in range(img.shape[0]):
        ch = img[c].float()
        m = ch != 0
        if m.sum() < 10:
            m = torch.ones_like(m)
        mu, sd = ch[m].mean(), ch[m].std()
        out[c] = torch.where(m, (ch - mu) / (sd + eps), torch.zeros_like(ch))
    return out


def foreground_bbox(img: torch.Tensor):
    nz = (img != 0).any(0).nonzero()
    if len(nz) == 0:
        return [(0, s) for s in img.shape[1:]]
    lo = nz.min(0).values.tolist()
    hi = (nz.max(0).values + 1).tolist()
    return list(zip(lo, hi))


def crop_or_pad(img, label, size, center=None, bbox=None):
    """Crop / pad (K, *S) and (*S) to ``size`` around ``center`` (voxel coords)."""
    sd = img.dim() - 1
    shape = img.shape[1:]
    if center is None:
        center = [(lo + hi) // 2 for lo, hi in (bbox or [(0, s) for s in shape])]
    starts = []
    for d in range(sd):
        s = int(center[d]) - size[d] // 2
        s = max(min(s, shape[d] - size[d]), 0) if shape[d] >= size[d] else 0
        starts.append(s)
    sl = tuple(slice(st, st + size[d]) for d, st in enumerate(starts))
    img = img[(slice(None),) + sl]
    label = label[sl] if label is not None else None
    pads = []
    for d in reversed(range(sd)):
        extra = size[d] - img.shape[1 + d]
        pads += [extra // 2, extra - extra // 2]
    if any(pads):
        img = F.pad(img, pads)
        if label is not None:
            label = F.pad(label, pads)
    return img, label
