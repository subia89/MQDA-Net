#!/usr/bin/env python
"""Generate structured reports and score them (BLEU, ROUGE-1/L, METEOR, BERTScore, RadGraph-F1).

python scripts/generate_reports.py --run runs/brats2023_joint [--split splits/brats2023_report_test.txt]

Loads ``best.pt`` (vision, graph, Q-Former, alignment heads) and the LoRA
adapter saved next to it. Writes ``reports.jsonl``, ``text_metrics.json`` and
``per_report.csv``.
"""
import argparse
import csv
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mqda.builder import build_model  # noqa: E402
from mqda.data.build import build_datasets  # noqa: E402
from mqda.data.datasets import collate  # noqa: E402
from mqda.eval.metrics import text_metrics  # noqa: E402
from mqda.utils.checkpoint import load_checkpoint  # noqa: E402
from mqda.utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with best.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None, help="override data.val_split")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--no-radgraph", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    ckpt_path = os.path.join(args.run, "best.pt")
    cfg = load_config(args.config, args.overrides) if args.config else \
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    if args.split:
        cfg["data"]["val_split"] = args.split
    cfg["model"]["blip2_init"] = None          # weights come from the checkpoint
    model = build_model(cfg, with_report=True)
    load_checkpoint(ckpt_path, model)
    model.to(args.device).eval()

    _, ds = build_datasets(cfg["data"], model.cfg)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    records = []
    amp = torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                         enabled=args.device.startswith("cuda"))
    with amp:
        for batch in tqdm(dl):
            out = model.generate_reports(batch["image"].to(args.device), notes=batch["notes"],
                                         max_new_tokens=args.max_new_tokens)
            for j, cid in enumerate(batch["case_id"]):
                records.append({"case_id": cid, "prediction": out["reports"][j].strip(),
                                "reference": batch["report"][j],
                                "prompt_facts": out["measurements"][j].as_text()})
    with open(os.path.join(args.run, "reports.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scored = [r for r in records if r["reference"]]
    if scored:
        m = text_metrics([r["prediction"] for r in scored], [r["reference"] for r in scored],
                         use_bertscore=not args.no_bertscore, use_radgraph=not args.no_radgraph,
                         per_sample=True)
        per = m.pop("per_sample")
        with open(os.path.join(args.run, "text_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        with open(os.path.join(args.run, "per_report.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", *per.keys()])
            for i, r in enumerate(scored):
                w.writerow([r["case_id"], *[per[k][i] for k in per]])
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
