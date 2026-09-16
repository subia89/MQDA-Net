#!/usr/bin/env python
"""Module-wise parameter count (cf. paper Table 15).

python scripts/count_params.py --config configs/base.yaml [--with-llm]
Without --with-llm the language model is not downloaded; the LoRA and
Q-Former sizes are computed for Llama-3.2-11B dimensions (hidden 4096,
40 decoder layers of which 32 are self-attention, 8 KV heads).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqda.builder import build_model  # noqa: E402
from mqda.models.report import QFormerProjector  # noqa: E402
from mqda.utils.config import load_config  # noqa: E402


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6


def lora_params(hidden=4096, kv_dim=1024, layers=32, rank=16, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    dims = {"q_proj": (hidden, hidden), "k_proj": (hidden, kv_dim),
            "v_proj": (hidden, kv_dim), "o_proj": (hidden, hidden)}
    return sum(rank * (i + o) for t in targets for i, o in [dims[t]]) * layers / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["model"]["blip2_init"] = None
    model = build_model(cfg, with_report=args.with_llm)
    c = model.cfg
    rows = [("Encoder + MSASPP", count(model.encoder)),
            ("Dual-branch decoder + DFCAM", count(model.decoder)),
            ("Quantum head (total)", count(model.quantum_head)),
            ("  of which circuit parameters", model.quantum_head.quantum_parameter_count() / 1e6),
            ("Knowledge graph + GAT", count(model.graph) + count(model.graph_token) if model.graph else 0.0)]
    if args.with_llm:
        rows += [("Q-Former projector", count(model.qformer)),
                 ("LoRA + alignment heads", count(model.report))]
    else:
        q = QFormerProjector(c.base_channels * 16, c.gat_dim, c.n_qubits, 4096, c.n_vis_queries,
                             c.n_graph_queries, c.n_quant_queries, c.qformer_layers,
                             c.qformer_hidden, c.qformer_encoder_hidden, c.qformer_cross_freq)
        llm = cfg.get("llm", {})
        rows += [("Q-Former projector (Llama-11B dims)", count(q)),
                 ("LoRA (Llama-11B dims, estimate)",
                  lora_params(rank=llm.get("lora_rank", 16),
                              targets=tuple(llm.get("lora_target_modules",
                                                    ["q_proj", "k_proj", "v_proj", "o_proj"]))))]
    total = sum(v for k, v in rows if not k.startswith("  "))
    width = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        if k.startswith("  "):
            print(f"{k:<{width}}{int(round(v * 1e6)):10d}")
        else:
            print(f"{k:<{width}}{v:10.2f} M")
    print(f"{'Total trainable':<{width}}{total:10.2f} M")


if __name__ == "__main__":
    main()
