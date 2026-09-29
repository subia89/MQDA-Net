# v1.0.0 — configuration corresponding to the published results

This release is the state of the repository that corresponds to the protocol
reported in the MQDA-Net manuscript (IJIES, Paper ID 20265965). Cite this tag,
or its commit hash, when reproducing the reported experiments.

## Correspondence with Table 1 of the manuscript

| Table 1 | Repository |
|---|---|
| Input resolution 3D / 2D — 128×128×128 / 224×224 | `data.size` in `configs/brats2023_*.yaml` and `configs/figshare_br35h.yaml` |
| Backbone — MedNeXt-L | `model.enc_blocks`, `model.enc_exp` in `configs/base.yaml` |
| MSASPP dilation rates — {1, 2, 4, 6} | `model.aspp_rates: [1, 2, 4, 6]` |
| DFCAM embedding dimension — 256 | `model.dfcam_dim: 256` |
| Qubits / PQC layers — 12 / 2 | `model.n_qubits: 12`, `model.pqc_layers: 2` |
| Parallel quantum kernels — 4 | `model.n_kernels: 4` |
| GAT layers / heads — 3 / 4 | `model.gat_layers: 3`, `model.gat_heads: 4` |
| LoRA rank / alpha — 16 / 32 | `llm.lora_rank: 16`, `llm.lora_alpha: 32` |
| Optimizer — AdamW | `mqda/engine.py` |
| Learning rate — 1e-4 / 2e-5 | `train.lr`, `train.llm_lr` |
| Weight decay — 1e-5 | `train.weight_decay` |
| Epochs (seg + cls / LLM) — 200 / 50 | `train.epochs` in `configs/base.yaml` and `configs/brats2023_joint.yaml` |
| Batch size 3D / 2D / LLM — 4 / 32 / 8 | `configs/base.yaml`, `configs/figshare_br35h.yaml`, `configs/brats2023_joint.yaml` |
| Loss weights λ1–λ4 — 1.0, 0.5, 1.0, 0.1 | `train.lambdas: [1.0, 0.5, 1.0, 0.1]` |
| Label smoothing / focal γ — 0.1 / 2.0 | `train.label_smoothing`, `train.focal_gamma` |
| Early stopping patience — 20 epochs | `train.patience: 20` |
| Quantum simulator — PennyLane 0.35, default.qubit | `model.q_backend: pennylane`, `requirements.txt` |

## Datasets

The classification benchmark is the Nickparvar Brain Tumor MRI Dataset,
**version 1**: 7,023 images (glioma 1,621, meningioma 1,645, pituitary 1,757,
no tumor 2,000). Version 2 of that Kaggle release is class-balanced at 7,200
images and is **not** the version the manuscript reports; select version 1
explicitly when downloading.

The manuscript reports an image-level stratified 80/20 split of the full 7,023
images (5,618 train / 1,405 test) rather than the Training/Testing folders that
ship with the dataset (5,712 / 1,311). The split indices are listed under
`splits/`.

## Changes in this release

- Stage-1 epochs corrected from 300 to 200 in `configs/base.yaml`,
  `configs/brats2023_vision.yaml`, `configs/brats2024_finetune.yaml`,
  `configs/figshare_br35h.yaml`, `README.md` and `docs/IMPLEMENTATION_NOTES.md`,
  to match Table 1.
- `model.q_backend` default set to `pennylane`, matching the simulator named in
  Table 1. The built-in `torch` simulator remains available.
- The classification benchmark is named consistently as the Nickparvar
  composite; `configs/kaggle_7k.yaml` is renamed `configs/nickparvar.yaml`.
- Removed a stale reference to an earlier draft of the manuscript.

## Not included

Trained weights, the 150 clinician-annotated reports and the 50 curated DPO
preference pairs are not distributed here. They are available from the
corresponding author on reasonable request.
