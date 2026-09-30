"""Training and validation loops (Accelerate; DeepSpeed ZeRO-2 via accelerate config)."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.build import build_datasets
from .data.datasets import collate
from .eval.metrics import classification_metrics, segmentation_metrics
from .losses.total import MQDALoss
from .utils.checkpoint import load_checkpoint, save_checkpoint

log = logging.getLogger("mqda")

DEFAULT_TRAIN = {
    "stage": "vision", "epochs": 200, "batch_size": 4, "lr": 1e-4, "llm_lr": 2e-5,
    "weight_decay": 1e-5, "lambdas": [1.0, 0.5, 1.0, 0.1], "w_binary": 0.8,
    "w_semantic": 1.0, "focal_gamma": 2.0, "label_smoothing": 0.1, "patience": 20,
    "precision": "bf16", "grad_clip": 1.0, "grad_accum": 1, "scheduler": "cosine",
    "warmup_epochs": 0, "num_workers": 4, "init_from": None, "resume": None,
    "val_every": 1, "max_val_batches": None, "monitor": "auto", "log_every": 20,
    "seed": 42, "batch_size_2d": 32, "freeze_vision": False,
}


def freeze_vision_modules(model):
    """Report stage of Table 2: the shared encoder, MSASPP, DFCAM, the decoder
    and the quantum head are evaluated but not updated; only the graph
    reasoner, the Q-Former projector and the LoRA adapters are trained."""
    frozen = 0
    for mod in (model.encoder, model.decoder, model.quantum_head):
        for p in mod.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    model.encoder.eval()
    model.decoder.eval()
    model.quantum_head.eval()
    model._frozen_vision = True
    log.info("froze %.2f M vision parameters for the report stage", frozen / 1e6)
    return frozen


def set_train_mode(model):
    """model.train() that keeps the frozen vision modules of the report stage in
    eval mode, so their normalisation statistics are not updated either."""
    model.train()
    core = model.module if hasattr(model, "module") else model
    if getattr(core, "_frozen_vision", False):
        core.encoder.eval()
        core.decoder.eval()
        core.quantum_head.eval()


def make_optimizer(model, tcfg):
    vision = [p for p in model.vision_parameters() if p.requires_grad]
    groups = [{"params": vision, "lr": tcfg["lr"], "name": "vision"}]
    lang = list(model.language_parameters())
    if lang:
        groups.append({"params": lang, "lr": tcfg["llm_lr"], "name": "language"})
    return torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])


def make_scheduler(opt, tcfg, steps_per_epoch):
    total = tcfg["epochs"] * steps_per_epoch
    warm = tcfg["warmup_epochs"] * steps_per_epoch
    if tcfg["scheduler"] == "none":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def f(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def validate(model, loader, criterion, accelerator, class_names, with_report, max_batches=None,
             distances=False):
    model.eval()
    seg_scores, y_true, y_pred = [], [], []
    losses = defaultdict(list)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with accelerator.autocast():
            out = model(batch["image"], seg_labels=batch["seg"],
                        reports=batch["report"] if with_report else None,
                        notes=batch["notes"], with_report=with_report)
            _, logs = criterion(out, batch["seg"], batch["cls"], batch["has_mask"])
        for k, v in logs.items():
            losses[k].append(float(v))
        pred = out["semantic_logits"].argmax(1).cpu().numpy()
        gt = batch["seg"].cpu().numpy()
        for j in range(len(pred)):
            if bool(batch["has_mask"][j]):
                seg_scores.append(segmentation_metrics(pred[j], gt[j], distances=distances))
        y_true += batch["cls"].tolist()
        y_pred += out["cls_logits"].argmax(-1).tolist()
    metrics = {f"val_{k}": float(np.mean(v)) for k, v in losses.items()}
    if seg_scores:
        for k in seg_scores[0]:
            metrics[k] = float(np.mean([s[k] for s in seg_scores]))
    if any(t >= 0 for t in y_true):
        cm = classification_metrics(y_true, y_pred, class_names)
        metrics.update({k: v for k, v in cm.items() if k != "confusion_matrix"})
    set_train_mode(model)
    return metrics


def monitor_value(metrics, monitor):
    if monitor == "auto":
        score = 0.0
        if "dice_mean" in metrics:
            score += metrics["dice_mean"]
        if "accuracy" in metrics:
            score += metrics["accuracy"]
        if "val_loss_txt" in metrics and metrics["val_loss_txt"] > 0:
            score -= 0.1 * metrics["val_loss_txt"]
        return score
    if monitor.startswith("-"):
        return -metrics[monitor[1:]]
    return metrics[monitor]


def train(cfg: dict, model, accelerator):
    tcfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    out_dir = cfg.get("output_dir", "runs/mqda")
    os.makedirs(out_dir, exist_ok=True)
    with_report = model.report is not None
    class_names = model.cfg.class_names

    if tcfg["freeze_vision"]:
        freeze_vision_modules(model)
    if cfg["data"].get("type") == "alternating":
        return train_alternating(cfg, tcfg, model, accelerator)

    train_ds, val_ds = build_datasets(cfg["data"], model.cfg)
    log.info("train %d / val %d samples", len(train_ds), len(val_ds))
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg["num_workers"], collate_fn=collate,
                          drop_last=len(train_ds) > tcfg["batch_size"],
                          pin_memory=torch.cuda.is_available(), persistent_workers=tcfg["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                        num_workers=tcfg["num_workers"], collate_fn=collate)

    if tcfg["init_from"]:
        load_checkpoint(tcfg["init_from"], model)
        log.info("initialised from %s", tcfg["init_from"])

    criterion = MQDALoss(tuple(tcfg["lambdas"]), tcfg["w_binary"], tcfg["w_semantic"],
                         tcfg["focal_gamma"], tcfg["label_smoothing"])
    opt = make_optimizer(model, tcfg)
    steps_per_epoch = max(len(train_dl) // tcfg["grad_accum"], 1)
    sched = make_scheduler(opt, tcfg, steps_per_epoch)
    start_epoch = 0
    if tcfg["resume"]:
        ck = load_checkpoint(tcfg["resume"], model, opt)
        start_epoch = (ck.get("epoch") or -1) + 1
        for _ in range(start_epoch * steps_per_epoch):
            sched.step()

    model, opt, train_dl, val_dl, sched = accelerator.prepare(model, opt, train_dl, val_dl, sched)
    best, bad_epochs = -float("inf"), 0
    history_path = os.path.join(out_dir, "history.jsonl")
    step = 0
    for epoch in range(start_epoch, tcfg["epochs"]):
        set_train_mode(model)
        t0 = time.time()
        running = defaultdict(float)
        n = 0
        for batch in train_dl:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    out = model(batch["image"], seg_labels=batch["seg"],
                                reports=batch["report"] if with_report else None,
                                notes=batch["notes"], with_report=with_report)
                    loss, logs = criterion(out, batch["seg"], batch["cls"], batch["has_mask"])
                accelerator.backward(loss)
                if accelerator.sync_gradients and tcfg["grad_clip"]:
                    accelerator.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            for k, v in logs.items():
                running[k] += float(v)
            n += 1
            step += 1
            if step % tcfg["log_every"] == 0:
                log.info("epoch %d step %d loss %.4f", epoch, step, running["loss"] / n)
        record = {"epoch": epoch, "time_s": round(time.time() - t0, 1),
                  **{f"train_{k}": v / max(n, 1) for k, v in running.items()}}
        if (epoch + 1) % tcfg["val_every"] == 0:
            record.update(validate(model, val_dl, criterion, accelerator, class_names,
                                   with_report, tcfg["max_val_batches"]))
            score = monitor_value(record, tcfg["monitor"])
            record["monitor"] = score
            core = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                save_checkpoint(os.path.join(out_dir, "last.pt"), core, opt, epoch, record, cfg)
            if score > best:
                best, bad_epochs = score, 0
                if accelerator.is_main_process:
                    save_checkpoint(os.path.join(out_dir, "best.pt"), core, None, epoch, record, cfg)
                    if with_report and hasattr(core.report.llm, "save_pretrained"):
                        core.report.llm.save_pretrained(os.path.join(out_dir, "lora_adapter"))
            else:
                bad_epochs += 1
        if accelerator.is_main_process:
            log.info(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                 for k, v in record.items()}))
            with open(history_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if bad_epochs >= tcfg["patience"]:
            log.info("early stopping after %d epochs without improvement", bad_epochs)
            break
    return best


# --------------------------------------------------------------------------
# Stage 1 with two cohorts (paper Sec. 3.7, Table 2 and Algorithm 1)
# --------------------------------------------------------------------------

@torch.no_grad()
def validate_alternating(model, seg_dl, cls_dl, criterion, accelerator, class_names,
                         max_batches=None, distances=False):
    model.eval()
    metrics, seg_scores, losses = {}, [], defaultdict(list)
    for i, batch in enumerate(seg_dl):
        if max_batches and i >= max_batches:
            break
        with accelerator.autocast():
            out = model(batch["image"], task="seg")
            _, logs = criterion(out, batch["seg"], None, batch["has_mask"])
        for k, v in logs.items():
            losses[k].append(float(v))
        pred = out["semantic_logits"].argmax(1).cpu().numpy()
        gt = batch["seg"].cpu().numpy()
        for j in range(len(pred)):
            if bool(batch["has_mask"][j]):
                seg_scores.append(segmentation_metrics(pred[j], gt[j], distances=distances))
    y_true, y_pred = [], []
    for i, batch in enumerate(cls_dl):
        if max_batches and i >= max_batches:
            break
        with accelerator.autocast():
            out = model(batch["image"], task="cls")
            _, logs = criterion(out, None, batch["cls"])
        for k, v in logs.items():
            losses[k].append(float(v))
        y_true += batch["cls"].tolist()
        y_pred += out["cls_logits"].argmax(-1).tolist()
    metrics.update({f"val_{k}": float(np.mean(v)) for k, v in losses.items()})
    if seg_scores:
        for k in seg_scores[0]:
            metrics[k] = float(np.mean([s[k] for s in seg_scores]))
    if any(t >= 0 for t in y_true):
        cm = classification_metrics(y_true, y_pred, class_names)
        metrics.update({k: v for k, v in cm.items() if k != "confusion_matrix"})
    set_train_mode(model)
    return metrics


def train_alternating(cfg, tcfg, model, accelerator):
    """One epoch presents both cohorts in alternating task-specific steps: a
    segmentation step (3D BraTS batch, L_seg, no quantum head) and a
    classification step (2D batch through A_2D->3D, L_cls, no decoder). Both
    update the shared encoder and MSASPP."""
    out_dir = cfg.get("output_dir", "runs/mqda")
    class_names = model.cfg.class_names
    data = cfg["data"]
    seg_tr, seg_va = build_datasets({**data["segmentation"], "augment": data.get("augment", {})},
                                    model.cfg)
    cls_tr, cls_va = build_datasets({**data["classification"], "augment": data.get("augment", {})},
                                    model.cfg)
    log.info("segmentation %d/%d, classification %d/%d samples",
             len(seg_tr), len(seg_va), len(cls_tr), len(cls_va))
    # the last, smaller batch is kept, so an epoch has ceil(N / batch) steps of each
    # type: 251 segmentation + 176 classification steps for 1,001 volumes / 5,618 images
    mk = lambda ds, bs, sh: DataLoader(  # noqa: E731
        ds, batch_size=bs, shuffle=sh, num_workers=tcfg["num_workers"], collate_fn=collate,
        drop_last=False, pin_memory=torch.cuda.is_available())
    seg_dl, cls_dl = mk(seg_tr, tcfg["batch_size"], True), mk(cls_tr, tcfg["batch_size_2d"], True)
    seg_vl, cls_vl = mk(seg_va, tcfg["batch_size"], False), mk(cls_va, tcfg["batch_size_2d"], False)

    if tcfg["init_from"]:
        load_checkpoint(tcfg["init_from"], model)
    criterion = MQDALoss(tuple(tcfg["lambdas"]), tcfg["w_binary"], tcfg["w_semantic"],
                         tcfg["focal_gamma"], tcfg["label_smoothing"])
    opt = make_optimizer(model, tcfg)
    steps_per_epoch = max(len(seg_dl) + len(cls_dl), 1)
    sched = make_scheduler(opt, tcfg, steps_per_epoch)
    model, opt, seg_dl, cls_dl, seg_vl, cls_vl, sched = accelerator.prepare(
        model, opt, seg_dl, cls_dl, seg_vl, cls_vl, sched)

    best, bad_epochs, step = -float("inf"), 0, 0
    history_path = os.path.join(out_dir, "history.jsonl")
    for epoch in range(tcfg["epochs"]):
        set_train_mode(model)
        t0 = time.time()
        running, counts = defaultdict(float), defaultdict(int)
        seg_it, cls_it = iter(seg_dl), iter(cls_dl)
        n_seg, n_cls = len(seg_dl), len(cls_dl)
        # interleave the two step types across the epoch
        plan = []
        for i in range(max(n_seg, n_cls)):
            if i < n_seg:
                plan.append("seg")
            if i < n_cls:
                plan.append("cls")
        for task in plan:
            batch = next(seg_it) if task == "seg" else next(cls_it)
            with accelerator.autocast():
                out = model(batch["image"], task=task)
                if task == "seg":
                    loss, logs = criterion(out, batch["seg"], None, batch["has_mask"])
                else:
                    loss, logs = criterion(out, None, batch["cls"])
            accelerator.backward(loss)
            if tcfg["grad_clip"]:
                accelerator.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            for k, v in logs.items():
                running[f"{task}_{k}"] += float(v)
            counts[task] += 1
            step += 1
            if step % tcfg["log_every"] == 0:
                log.info("epoch %d step %d (%s) loss %.4f", epoch, step, task, float(logs["loss"]))
        record = {"epoch": epoch, "time_s": round(time.time() - t0, 1),
                  "seg_steps": counts["seg"], "cls_steps": counts["cls"],
                  **{f"train_{k}": v / max(counts[k.split('_')[0]], 1) for k, v in running.items()}}
        if (epoch + 1) % tcfg["val_every"] == 0:
            record.update(validate_alternating(model, seg_vl, cls_vl, criterion, accelerator,
                                               class_names, tcfg["max_val_batches"]))
            score = monitor_value(record, tcfg["monitor"])
            record["monitor"] = score
            core = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                save_checkpoint(os.path.join(out_dir, "last.pt"), core, opt, epoch, record, cfg)
                if score > best:
                    save_checkpoint(os.path.join(out_dir, "best.pt"), core, None, epoch, record, cfg)
            bad_epochs = 0 if score > best else bad_epochs + 1
            best = max(best, score)
        if accelerator.is_main_process:
            log.info(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                 for k, v in record.items()}))
            with open(history_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if bad_epochs >= tcfg["patience"]:
            log.info("early stopping after %d epochs without improvement", bad_epochs)
            break
    return best
