#!/usr/bin/env python
"""Train MQDA-Net.

Stage 1 (vision, configs/stage1_alternating.yaml): encoder + dual-branch decoder
                  + quantum head, alternating segmentation (L_seg) and
                  classification (L_cls) steps over the shared encoder; 200 epochs.
Stage 2 (joint, configs/brats2023_joint.yaml): encoder, MSASPP, DFCAM, decoder and
                  quantum head frozen (train.freeze_vision: true); only the graph
                  reasoner, the Q-Former projector and the LoRA adapters are trained,
                  with L_txt + L_align (Table 2 of the manuscript); 50 epochs.

Examples
--------
python scripts/train.py --config configs/stage1_alternating.yaml
accelerate launch --config_file configs/accelerate_zero2.yaml \
    scripts/train.py --config configs/brats2023_joint.yaml \
    train.init_from=runs/stage1_alternating/best.pt
python scripts/train.py --config configs/smoke_test.yaml      # synthetic data, CPU
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs, set_seed  # noqa: E402

from mqda.builder import build_model  # noqa: E402
from mqda.engine import DEFAULT_TRAIN, train  # noqa: E402
from mqda.utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tcfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}

    precision = {"bf16": "bf16", "fp16": "fp16", "fp32": "no", "no": "no"}[tcfg["precision"]]
    accelerator = Accelerator(
        mixed_precision=precision, gradient_accumulation_steps=tcfg["grad_accum"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    logging.basicConfig(level=logging.INFO if accelerator.is_main_process else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(tcfg["seed"])
    out_dir = cfg.setdefault("output_dir", f"runs/{cfg.get('experiment', 'mqda')}")
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    model = build_model(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger("mqda").info("trainable parameters: %.2f M", n_train / 1e6)
    best = train(cfg, model, accelerator)
    logging.getLogger("mqda").info("best monitor value: %.4f", best)


if __name__ == "__main__":
    main()
