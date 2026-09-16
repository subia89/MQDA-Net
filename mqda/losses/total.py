"""Composite objective L_total = l1 L_seg + l2 L_cls + l3 L_txt + l4 L_align (Eq. 2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .classification import FocalLabelSmoothingLoss
from .segmentation import SegmentationLoss


class MQDALoss(nn.Module):
    def __init__(self, lambdas=(1.0, 0.5, 1.0, 0.1), w_binary=0.8, w_semantic=1.0,
                 focal_gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.l1, self.l2, self.l3, self.l4 = lambdas
        self.seg = SegmentationLoss(w_binary, w_semantic)
        self.cls = FocalLabelSmoothingLoss(focal_gamma, label_smoothing)

    def forward(self, out, seg_labels=None, cls_labels=None, has_mask=None):
        dev = out["cls_logits"].device
        zero = torch.zeros((), device=dev)
        logs = {}
        l_seg = zero
        if seg_labels is not None:
            l_seg, parts = self.seg(out["binary_logits"].float(), out["semantic_logits"].float(),
                                    seg_labels, has_mask)
            logs.update(parts)
        l_cls = self.cls(out["cls_logits"], cls_labels) if cls_labels is not None else zero
        l_txt = out.get("loss_txt", zero)
        l_align = out.get("loss_align", zero)
        total = self.l1 * l_seg + self.l2 * l_cls + self.l3 * l_txt + self.l4 * l_align
        logs.update(loss_seg=l_seg.detach(), loss_cls=l_cls.detach(),
                    loss_txt=l_txt.detach(), loss_align=l_align.detach(), loss=total.detach())
        return total, logs
