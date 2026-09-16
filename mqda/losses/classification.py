"""Focal cross-entropy with label smoothing (paper Eq. 12).

L_cls = - sum_a alpha_a (1 - p_a)^gamma log p_a,
alpha_a = 1 - beta for the target class and beta / (C - 1) otherwise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, beta: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0
        logits, target = logits[valid].float(), target[valid]
        c = logits.shape[-1]
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        alpha = torch.full_like(p, self.beta / max(c - 1, 1))
        alpha.scatter_(1, target.unsqueeze(1), 1.0 - self.beta)
        loss = -(alpha * (1 - p) ** self.gamma * logp).sum(-1)
        return loss.mean()
