"""Dataset construction from the ``data`` section of a YAML config."""
from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .datasets import BraTSDataset, FigShareDataset, FolderDataset, random_split_indices


class SyntheticDataset(Dataset):
    """Random volumes/slices with ellipsoidal NCR/ED/ET lesions and template
    reports. Only for smoke tests of the full pipeline - not for evaluation."""

    TYPES = ["glioma", "meningioma", "pituitary", "no tumor"]

    def __init__(self, n=16, spatial_dims=3, in_channels=4, size=None, n_classes=4, seed=0,
                 with_reports=True):
        self.n, self.sd, self.k = n, spatial_dims, in_channels
        self.size = tuple(size or ([64] * spatial_dims))
        self.n_classes = n_classes
        self.seed = seed
        self.with_reports = with_reports

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        cls = int(torch.randint(0, self.n_classes, (1,), generator=g))
        img = torch.randn(self.k, *self.size, generator=g) * 0.3
        seg = torch.zeros(self.size, dtype=torch.long)
        if cls != self.n_classes - 1:
            grids = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in self.size], indexing="ij")
            c = (torch.rand(self.sd, generator=g) - 0.5) * 0.8
            r = 0.2 + 0.15 * torch.rand(1, generator=g).item()
            dist = sum((grids[d] - c[d]) ** 2 for d in range(self.sd)).sqrt()
            seg[dist < r] = 2
            seg[dist < 0.6 * r] = 3
            seg[dist < 0.3 * r] = 1
            for lab, shift in ((1, -0.5), (2, 0.8), (3, 1.5)):
                img[:, seg == lab] += shift * (1 + 0.3 * cls)
        vol = int((seg > 0).sum())
        report = ""
        if self.with_reports:
            name = self.TYPES[cls] if cls < len(self.TYPES) else f"class {cls}"
            report = (f"Technique: synthetic MRI. Findings: lesion of {vol} voxels. "
                      f"Laterality and Lobe: unspecified. Impression: {name}. BT-RADS: 2.")
        return {"image": img, "seg": seg, "has_mask": True, "cls": cls, "report": report,
                "notes": "headache" if i % 2 else "", "case_id": f"synthetic_{i:04d}"}


def build_datasets(data_cfg: dict, model_cfg):
    t = data_cfg["type"]
    aug = data_cfg.get("augment", {})
    if t == "synthetic":
        common = dict(spatial_dims=data_cfg.get("spatial_dims", model_cfg.spatial_dims),
                      in_channels=data_cfg.get("in_channels", model_cfg.in_channels),
                      size=data_cfg.get("size"), n_classes=len(model_cfg.class_names))
        return (SyntheticDataset(data_cfg.get("n_train", 16), seed=0, **common),
                SyntheticDataset(data_cfg.get("n_val", 4), seed=1, **common))
    if t == "brats":
        kw = dict(root=data_cfg["root"], naming=str(data_cfg.get("naming", "2023")),
                  label_map=data_cfg.get("label_map"), size=data_cfg.get("size", [128] * 3),
                  reports_jsonl=data_cfg.get("reports_jsonl"),
                  class_index=model_cfg.class_names.index(data_cfg.get("class_name", "glioma")),
                  cache=data_cfg.get("cache", False))
        return (BraTSDataset(split_file=data_cfg.get("train_split"), train=True, augment=aug, **kw),
                BraTSDataset(split_file=data_cfg.get("val_split"), train=False, **kw))
    if t in ("figshare", "folder", "figshare+folder"):
        size = data_cfg.get("size", [224, 224])
        parts_train, parts_val = [], []
        test_frac = data_cfg.get("test_fraction", 0.2)
        seed = data_cfg.get("seed", 42)
        if t in ("figshare", "figshare+folder"):
            fs = data_cfg["figshare"]
            mk = lambda train, split: FigShareDataset(  # noqa: E731
                fs["root"], model_cfg.class_names, split_file=split, size=size, train=train,
                augment=aug, mask_label=fs.get("mask_label", 1))
            if fs.get("train_split"):
                parts_train.append(mk(True, fs["train_split"]))
                parts_val.append(mk(False, fs.get("val_split")))
            else:
                full_tr, full_va = mk(True, None), mk(False, None)
                tr, va = random_split_indices(len(full_tr), test_frac, seed)
                parts_train.append(Subset(full_tr, tr))
                parts_val.append(Subset(full_va, va))
        if t in ("folder", "figshare+folder"):
            fo = data_cfg["folder"]
            mk = lambda root, train: FolderDataset(  # noqa: E731
                root, fo["class_map"], size=size, train=train, augment=aug,
                mask_root=fo.get("mask_root"), mask_label=fo.get("mask_label", 1))
            if fo.get("train_root"):
                parts_train.append(mk(fo["train_root"], True))
                parts_val.append(mk(fo["val_root"], False))
            else:
                full_tr, full_va = mk(fo["root"], True), mk(fo["root"], False)
                tr, va = random_split_indices(len(full_tr), test_frac, seed)
                parts_train.append(Subset(full_tr, tr))
                parts_val.append(Subset(full_va, va))
        return ConcatDataset(parts_train), ConcatDataset(parts_val)
    raise ValueError(f"unknown data type {t!r}")
