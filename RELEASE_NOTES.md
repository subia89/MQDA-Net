# v1.0.1 — release cited by the manuscript

This release is the reference implementation of the protocol reported in the
MQDA-Net manuscript (IJIES, Paper ID 20265965). Cite this tag, or its commit
hash, when referring to the code.

## Changes in v1.0.1

- **Default epoch count.** The in-code default in `mqda/engine.py` was still
  300; it is now 200, matching Table 1 and every configuration file.
- **Frozen modules stay frozen.** In the report stage the encoder, decoder and
  quantum head now remain in evaluation mode throughout training
  (`set_train_mode`), so their normalisation statistics are not updated
  either; previously `model.train()` at the start of each epoch put them back
  into training mode. `tests/test_protocol.py` checks this.
- **Reference split, stated as such.** `splits/nickparvar_{train,test}.txt`
  were described as the split used for the reported results. They are a split
  regenerated with `scripts/make_splits.py --stratified-folder` (seed 42) that
  has the same sizes as the reported split — 5,618 / 1,405, with 324 / 329 /
  352 / 400 test images per class, the row totals of the confusion matrix in Figure 6 — but the lists
  of the reported experiment were not retained, so image membership is not
  necessarily the same. The README and `splits/README.md` now say so.
- **Table 3b on the manuscript's split protocol.** `scripts/overlap_analysis.py`
  now reports the expected train–test overlap (byte-identical, pixel-identical,
  zero-distance perceptual hash) over 200 random draws of the image-level
  stratified 80/20 split (n = 1,405), and the cross-class false-match rate of
  the perceptual hash over the pooled cohort; the partition as distributed with
  the dataset (5,712 / 1,311) is printed for reference only. `--test-list`
  reports the overlap of a given partition; `--cache` stores per-image hashes.
  `docs/table3b_overlap_manifest.json` is regenerated accordingly.
- **Stage I reads the manuscript's split.** `configs/stage1_alternating.yaml`
  now takes the classification cohort from `splits/nickparvar_{train,test}.txt`
  (5,618 / 1,405) instead of the dataset's own Training / Testing folders
  (5,712 / 1,311); `configs/brats2023_joint.yaml` and
  `configs/brats2024_finetune.yaml` initialise from the Stage I checkpoint
  (`runs/stage1_alternating/best.pt`), and the last, smaller batch of each
  cohort is kept, so an epoch has the 251 + 176 steps stated in the manuscript.
- **Documentation.** `scripts/train.py` no longer describes Stage 2 as training
  "everything under the four-term objective"; the Stage 2 configuration header no
  longer says "all four objectives"; the README describes `mqda/eval/stats.py` as descriptive
  bootstrap intervals (§5.1 of the manuscript reports no significance test);
  trained weights are stated as not distributed.

## Repository history relative to the manuscript

- `305e331` (16 September 2026, initial reference implementation): the README
  and the configuration files gave 300 Stage-I epochs, and the second stage
  (`configs/brats2023_joint.yaml`) trained every module under all four loss
  terms. This state does not implement the protocol of Table 2.
- `47f9f76` (29 September 2026, the commit first tagged v1.0.0): configuration
  files and README corrected to 200 Stage-I epochs; the in-code default
  remained 300 and the second stage still trained all modules.
- `2a1a0ec` to `ddcd9aa` (29–30 September 2026): alternating Stage-I loop,
  A_2D→3D input adaptation, `train.freeze_vision` for the report stage,
  explicit split lists and the Table 3b manifest; the v1.0.0 tag was moved to
  `ddcd9aa`.
- v1.0.1 (this release): in-code default 200; frozen modules kept in
  evaluation mode across epochs; split lists stated as a reference split;
  Table 3b on the manuscript's split protocol; documentation. This is the
  release the manuscript cites, by tag and by commit hash.

---

# v1.0.0 — configuration corresponding to the published results

Superseded by v1.0.1.

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
ship with the dataset (5,712 / 1,311). A reference split with those sizes is
committed as `splits/nickparvar_train.txt` and `splits/nickparvar_test.txt`
(regenerated with seed 42; see v1.0.1 above), and `scripts/overlap_analysis.py`
states Table 3b on the manuscript's split protocol
(`docs/table3b_overlap_manifest.json`).

## Training protocol (Table 2)

Stage 1 alternates a segmentation step (4 BraTS volumes, L_seg) with a
classification step (32 2D images, L_cls) over one shared MedNeXt-L encoder;
the 2D cohort reaches that encoder through the parameter-free A_2D→3D
adaptation and is pooled by global average pooling. Stage 2 freezes the
encoder, MSASPP, DFCAM, decoder and quantum head and trains only the graph
reasoner, the Q-Former and the LoRA adapters — `configs/stage1_alternating.yaml`
and `train.freeze_vision: true` in `configs/brats2023_joint.yaml`.

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
- Added the alternating two-cohort stage-1 loop, the A_2D→3D input adaptation
  and the global-average-pooling classification path, so the code follows
  Table 2 of the manuscript step by step; stage 2 now freezes the vision
  modules.
- Br35H is described as 3,000 slices (1,500 tumor / 1,500 tumor-free),
  matching the revised manuscript.

## Not included

Trained weights, the 150 clinician-annotated reports and the 50 curated DPO
preference pairs are not distributed here.
