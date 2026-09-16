"""Sliding-window segmentation for full-size volumes."""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_segment(model, image: torch.Tensor, roi, overlap=0.5, batch_size=2):
    """image (1, K, *S) -> semantic probabilities (1, C, *S), Gaussian-weighted."""
    sd = image.dim() - 2
    shape = image.shape[2:]
    pads = []
    for d in reversed(range(sd)):
        extra = max(roi[d] - shape[d], 0)
        pads += [0, extra]
    x = F.pad(image, pads)
    full = x.shape[2:]
    starts = []
    for d in range(sd):
        step = max(int(roi[d] * (1 - overlap)), 1)
        s = list(range(0, max(full[d] - roi[d], 0) + 1, step))
        if s[-1] != full[d] - roi[d]:
            s.append(full[d] - roi[d])
        starts.append(s)
    sigma = [r / 8 for r in roi]
    g = 1
    for d in range(sd):
        ax = torch.arange(roi[d], device=image.device) - (roi[d] - 1) / 2
        w = torch.exp(-(ax ** 2) / (2 * sigma[d] ** 2))
        g = g * w.view(*[-1 if i == d else 1 for i in range(sd)])
    g = (g / g.max()).clamp_min(1e-3)   # keep window corners from vanishing
    probs = None
    weight = torch.zeros(full, device=image.device)
    windows = list(itertools.product(*starts))
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        crops = torch.cat([x[(slice(None), slice(None)) + tuple(slice(s, s + r) for s, r in zip(w, roi))]
                           for w in chunk])
        feats = model.encoder(crops)
        p = model.decoder(feats)["semantic_logits"].float().softmax(1)
        if probs is None:
            probs = torch.zeros(1, p.shape[1], *full, device=image.device)
        for j, w in enumerate(chunk):
            sl = tuple(slice(s, s + r) for s, r in zip(w, roi))
            probs[(0, slice(None)) + sl] += p[j] * g
            weight[sl] += g
    probs = probs / weight.clamp_min(1e-6)
    crop = tuple(slice(0, s) for s in shape)
    return probs[(slice(None), slice(None)) + crop]
