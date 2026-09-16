#!/usr/bin/env python
"""Run MQDA-Net on one BraTS-style case directory.

python scripts/infer.py --run runs/brats2023_joint --case data/.../BraTS-GLI-00000-000 \
    [--notes "new onset seizures"] [--out outputs/BraTS-GLI-00000-000]

Writes the predicted segmentation (NIfTI, BraTS labels 1/2/3), the class
probabilities and, when the run has a language model, the structured report.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mqda.builder import build_model  # noqa: E402
from mqda.data.datasets import BRATS_MODALITIES  # noqa: E402
from mqda.data.transforms import crop_or_pad, foreground_bbox, zscore  # noqa: E402
from mqda.utils.checkpoint import load_checkpoint  # noqa: E402
from mqda.utils.inference import sliding_window_segment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--naming", default="2023", choices=list(BRATS_MODALITIES))
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_path = os.path.join(args.run, "best.pt")
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    cfg["model"]["blip2_init"] = None
    with_report = cfg.get("train", {}).get("stage") == "joint" and not args.no_report
    model = build_model(cfg, with_report=with_report)
    load_checkpoint(ckpt_path, model)
    model.to(args.device).eval()

    cid = os.path.basename(os.path.normpath(args.case))
    sep = "-" if args.naming == "2023" else "_"
    paths = [glob.glob(os.path.join(args.case, f"*{sep}{m}.nii*"))[0]
             for m in BRATS_MODALITIES[args.naming]]
    ref = nib.load(paths[0])
    vol = np.stack([np.asarray(nib.load(p).dataobj, dtype=np.float32) for p in paths])
    img = zscore(torch.from_numpy(vol)).to(args.device)
    size = cfg["data"].get("size", [128, 128, 128])

    with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                                         enabled=args.device.startswith("cuda")):
        probs = sliding_window_segment(model, img[None], size)
        labels = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
        crop, _ = crop_or_pad(img, None, size, bbox=foreground_bbox(img))
        if with_report:
            out = model.generate_reports(crop[None], notes=[args.notes])
        else:
            out = model.segment_and_classify(crop[None])
    out_dir = args.out or os.path.join("outputs", cid)
    os.makedirs(out_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(labels, ref.affine, ref.header), os.path.join(out_dir, f"{cid}-pred.nii.gz"))
    cls_p = out["cls_logits"].float().softmax(-1)[0].tolist()
    full_meas = model.measure(torch.from_numpy(labels.astype(np.int64)))[0]
    result = {"case_id": cid,
              "class_probabilities": dict(zip(model.cfg.class_names, cls_p)),
              "measurements": full_meas.as_text()}
    if with_report:
        result["report"] = out["reports"][0]
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
