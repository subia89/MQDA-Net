#!/usr/bin/env python
"""RadGraph-F1 Direct Preference Optimization refinement of the report generator.

Input: a JSONL of preference pairs, one per line:
  {"case_id": ..., "chosen": "<report>", "rejected": "<report>"}
or, to let RadGraph-F1 decide the preference against a reference,
  {"case_id": ..., "candidates": ["<report A>", "<report B>"], "reference": "<report>"}

python scripts/dpo_refine.py --run runs/brats2023_joint --pairs data/reports/dpo_pairs.jsonl

Only the LoRA adapter is updated; the vision modules provide the conditioning
tokens and stay frozen. The reference policy is the adapter as loaded
(its log-probabilities are cached before training).
"""
import argparse
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from mqda.builder import build_model  # noqa: E402
from mqda.data.build import build_datasets  # noqa: E402
from mqda.data.datasets import collate  # noqa: E402
from mqda.losses.preference import dpo_loss, sequence_logprob  # noqa: E402
from mqda.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402

log = logging.getLogger("dpo")


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "candidates" in r:
                from radgraph import F1RadGraph

                scorer = F1RadGraph(reward_level="partial")
                _, rewards, _, _ = scorer(hyps=r["candidates"], refs=[r["reference"]] * len(r["candidates"]))
                order = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
                r = {"case_id": r["case_id"], "chosen": r["candidates"][order[0]],
                     "rejected": r["candidates"][order[-1]]}
            pairs.append(r)
    return pairs


def seq_logp(model, tokens, prompt, report):
    rg = model.report
    t_vis, t_graph, t_quant = tokens
    prefix = torch.cat([t_vis, t_graph, t_quant], 1)
    emb, att, lab, _ = rg._assemble(prefix, [prompt], [report])
    out = rg.llm(inputs_embeds=emb, attention_mask=att, return_dict=True)
    return sequence_logprob(out.logits, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    ckpt_path = os.path.join(args.run, "best.pt")
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
    cfg["model"]["blip2_init"] = None
    model = build_model(cfg, with_report=True)
    load_checkpoint(ckpt_path, model)
    model.to(args.device)
    for p in model.vision_parameters():
        p.requires_grad_(False)

    pairs = load_pairs(args.pairs)
    # locate the images of the paired cases in the train + val datasets
    tr, va = build_datasets(cfg["data"], model.cfg)
    index = {}
    for ds in (tr, va):
        for i in range(len(ds)):
            cid = ds.ids[i] if hasattr(ds, "ids") else ds[i]["case_id"]
            index.setdefault(cid, (ds, i))
    pairs = [p for p in pairs if p["case_id"] in index]
    log.info("%d preference pairs with images", len(pairs))

    def conditioning(cid):
        ds, i = index[cid]
        item = collate([ds[i]])
        with torch.no_grad():
            model.eval()
            out = model(item["image"].to(args.device), notes=item["notes"], with_report=True)
        return tuple(t.detach() for t in out["tokens"]), out["prompts"][0]

    cache = {p["case_id"]: conditioning(p["case_id"]) for p in pairs}
    ref = []
    with torch.no_grad():
        for p in pairs:
            tok, prompt = cache[p["case_id"]]
            ref.append((seq_logp(model, tok, prompt, p["chosen"]),
                        seq_logp(model, tok, prompt, p["rejected"])))
    params = [p for p in model.report.llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.report.train()
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.shuffle(order)
        total = 0.0
        for k in order:
            p = pairs[k]
            tok, prompt = cache[p["case_id"]]
            pc = seq_logp(model, tok, prompt, p["chosen"])
            pr = seq_logp(model, tok, prompt, p["rejected"])
            loss, _ = dpo_loss(pc, pr, ref[k][0], ref[k][1], args.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        log.info("epoch %d  DPO loss %.4f", epoch, total / max(len(pairs), 1))
    out_dir = args.out or os.path.join(args.run, "dpo")
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(os.path.join(out_dir, "best.pt"), model, config=cfg)
    model.report.llm.save_pretrained(os.path.join(out_dir, "lora_adapter"))
    log.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()
