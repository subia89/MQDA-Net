"""Segmentation objective (paper Eqs. 7-8).

L_seg = w_b * BCE(M_b, Y_bin) + w_s * (DiceCE(M_s, Y_sem) + L_boundary)

DiceCE uses the squared-denominator soft Dice of Eq. 7. The boundary term
compares soft boundaries of the prediction and the one-hot target, where a
soft boundary is the morphological gradient (max-pool dilation minus
min-pool erosion) of the probability map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(N, *S) int -> (N, K, *S) float."""
    oh = F.one_hot(labels.long().clamp_min(0), num_classes)
    return oh.movedim(-1, 1).float()


def _max_pool(x, k=3):
    sd = x.dim() - 2
    pool = F.max_pool3d if sd == 3 else F.max_pool2d
    return pool(x, k, stride=1, padding=k // 2)


def soft_boundary(prob: torch.Tensor, k: int = 3) -> torch.Tensor:
    dil = _max_pool(prob, k)
    ero = -_max_pool(-prob, k)
    return dil - ero


def soft_dice_loss(prob, target, eps=1e-5, include_background=False):
    """1 - 2 sum(PY) / (sum(P^2) + sum(Y^2) + eps), averaged over classes."""
    if not include_background and prob.shape[1] > 1:
        prob, target = prob[:, 1:], target[:, 1:]
    dims = tuple(range(2, prob.dim()))
    inter = (prob * target).sum(dims)
    denom = (prob ** 2).sum(dims) + (target ** 2).sum(dims) + eps
    # eps is also added to the numerator so a class absent from both the
    # prediction and the target scores Dice = 1 instead of 0
    dice = (2 * inter + eps) / denom
    return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, eps=1e-5, include_background=False):
        super().__init__()
        self.eps = eps
        self.include_background = include_background

    def forward(self, logits, labels):
        k = logits.shape[1]
        ce = F.cross_entropy(logits, labels.long())
        prob = logits.softmax(1)
        return ce + soft_dice_loss(prob, one_hot(labels, k), self.eps, self.include_background)


class BoundaryLoss(nn.Module):
    """L_boundary = 1 - 2|dM ∩ dY| / (|dM| + |dY| + eps) on soft boundaries."""

    def __init__(self, eps=1e-5, kernel=3):
        super().__init__()
        self.eps = eps
        self.kernel = kernel

    def forward(self, logits, labels):
        k = logits.shape[1]
        prob = logits.softmax(1)[:, 1:]
        tgt = one_hot(labels, k)[:, 1:]
        bp, bt = soft_boundary(prob, self.kernel), soft_boundary(tgt, self.kernel)
        dims = tuple(range(2, bp.dim()))
        inter = (bp * bt).sum(dims)
        denom = bp.sum(dims) + bt.sum(dims) + self.eps
        return 1 - ((2 * inter + self.eps) / denom).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, w_binary=0.8, w_semantic=1.0, eps=1e-5):
        super().__init__()
        self.w_b = w_binary
        self.w_s = w_semantic
        self.dice_ce = DiceCELoss(eps)
        self.boundary = BoundaryLoss(eps)

    def forward(self, binary_logits, semantic_logits, labels, has_mask=None):
        """labels: (N, *S) with 0 = background. ``has_mask`` (N,) bool selects
        the samples that carry a ground-truth mask (e.g. Br35H has none)."""
        if has_mask is not None:
            if not bool(has_mask.any()):
                return semantic_logits.sum() * 0.0, {}
            binary_logits = binary_logits[has_mask]
            semantic_logits = semantic_logits[has_mask]
            labels = labels[has_mask]
        y_bin = (labels > 0).float().unsqueeze(1)
        l_bin = F.binary_cross_entropy_with_logits(binary_logits, y_bin)
        l_dce = self.dice_ce(semantic_logits, labels)
        l_bd = self.boundary(semantic_logits, labels)
        total = self.w_b * l_bin + self.w_s * (l_dce + l_bd)
        return total, {"seg_bce": l_bin.detach(), "seg_dicece": l_dce.detach(),
                       "seg_boundary": l_bd.detach()}
