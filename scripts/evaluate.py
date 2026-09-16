#!/usr/bin/env python
"""Evaluate segmentation (Dice, IoU, HD95, ASSD) and classification.

python scripts/evaluate.py --checkpoint runs/brats2023_vision/best.pt \
    [--config configs/brats2023_vision.yaml] [--full-volume] [--out results/brats2023]

Writes ``summary.json`` and ``per_case.csv`` (the per-case file feeds
scripts/significance.py).
"""
import argparse
import csv
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mqda.builder import build_model  # noqa: E402
from mqda.data.build import build_datasets  # noqa: E402
from mqda.data.datasets import collate  # noqa: E402
from mqda.data.transforms import crop_or_pad, foreground_bbox  # noqa: E402
from mqda.eval.metrics import classification_metrics, segmentation_metrics  # noqa: E402
from mqda.utils.checkpoint import load_checkpoint  # noqa: E402
from mqda.utils.config import load_config  # noqa: E402
from mqda.utils.inference import sliding_window_segment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", help="defaults to the config stored in the checkpoint")
    ap.add_argument("--out", default=None)
    ap.add_argument("--full-volume", action="store_true",
                    help="BraTS: sliding-window segmentation of the whole volume")
    ap.add_argument("--no-distances", action="store_true", help="skip HD95 / ASSD")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config, args.overrides) if args.config else ckpt["config"]
    model = build_model(cfg, with_report=False)
    load_checkpoint(args.checkpoint, model)
    model.to(args.device).eval()
    out_dir = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)

    _, val_ds = build_datasets(cfg["data"], model.cfg)
    full = args.full_volume and cfg["data"]["type"] == "brats"
    if full:
        val_ds.size = None
        args.batch_size = 1
    dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    size = cfg["data"].get("size", [128, 128, 128])
    rows, y_true, y_pred = [], [], []
    amp = torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                         enabled=args.device.startswith("cuda"))
    with torch.no_grad(), amp:
        for batch in tqdm(dl):
            img = batch["image"].to(args.device)
            if full:
                probs = sliding_window_segment(model, img, size, overlap=0.5)
                pred = probs.argmax(1).cpu().numpy()
                crop, _ = crop_or_pad(img[0], None, size, bbox=foreground_bbox(img[0]))
                logits = model.segment_and_classify(crop[None])["cls_logits"]
            else:
                out = model.segment_and_classify(img)
                pred = out["semantic_logits"].argmax(1).cpu().numpy()
                logits = out["cls_logits"]
            gt = batch["seg"].numpy()
            cls_pred = logits.argmax(-1).cpu().tolist()
            for j, cid in enumerate(batch["case_id"]):
                row = {"case_id": cid, "cls_true": int(batch["cls"][j]), "cls_pred": cls_pred[j]}
                if bool(batch["has_mask"][j]):
                    row.update(segmentation_metrics(pred[j], gt[j], distances=not args.no_distances))
                rows.append(row)
                y_true.append(row["cls_true"])
                y_pred.append(row["cls_pred"])

    summary = {}
    seg_rows = [r for r in rows if "dice_mean" in r]
    if seg_rows:
        for k in seg_rows[0]:
            if k.startswith(("dice", "iou", "hd95", "assd")):
                summary[k] = float(np.mean([r[k] for r in seg_rows]))
    if any(t >= 0 for t in y_true):
        summary.update(classification_metrics(y_true, y_pred, model.cfg.class_names))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
