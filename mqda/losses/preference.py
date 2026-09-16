"""Direct Preference Optimization loss for the RadGraph-F1 refinement step."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of token log-probabilities where labels != -100 (causal shift applied)."""
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    mask = labels != -100
    lp = torch.gather(logits.log_softmax(-1), 2, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """-log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)])."""
    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean(), margin.detach()
