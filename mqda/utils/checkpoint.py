"""Checkpoints that skip the frozen language-model weights."""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def trainable_state_dict(model: torch.nn.Module) -> dict:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    out = {}
    for k, v in model.state_dict().items():
        if k.startswith("report.llm.") and k not in trainable and "lora_" not in k:
            continue  # frozen backbone weights are reloaded from the hub
        out[k] = v.detach().cpu()
    return out


def save_checkpoint(path: str, model, optimizer=None, epoch=None, metrics=None, config=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": trainable_state_dict(model),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch, "metrics": metrics, "config": config}, path)


def load_checkpoint(path: str, model, optimizer=None, strict_vision=True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if not k.startswith("report.llm.")]
    if strict_vision:
        # a vision-stage checkpoint has no graph / Q-Former / report weights - that is fine
        core = [k for k in missing if k.startswith(("encoder.", "decoder.", "quantum_head."))]
        if core:
            raise RuntimeError(f"checkpoint is missing core weights: {core[:5]} ...")
    if missing:
        log.info("%d parameters initialised fresh (not in checkpoint), e.g. %s",
                 len(missing), missing[:3])
    if unexpected:
        log.warning("%d unexpected keys ignored, e.g. %s", len(unexpected), unexpected[:3])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
