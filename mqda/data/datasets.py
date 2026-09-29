"""Datasets used in the paper.

* :class:`BraTSDataset` - BraTS 2021 / 2023 (GLI) / 2024 post-treatment volumes.
* :class:`FigShareDataset` - Cheng et al. FigShare .mat slices with masks.
* :class:`FolderDataset` - image-folder datasets (Br35H, 3K-DS, 7K-DS, the
  Kaggle 7,023-image merge), optional masks in a parallel folder.

Every item is a dict with keys
``image`` (K, *S) float, ``seg`` (*S) long, ``has_mask`` bool, ``cls`` int
(-100 = unknown), ``report`` str, ``notes`` str, ``case_id`` str.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .transforms import Augment, crop_or_pad, foreground_bbox, zscore

BRATS_MODALITIES = {
    # BraTS 2023 / 2024 naming
    "2023": ["t1n", "t1c", "t2w", "t2f"],
    # BraTS 2021 naming
    "2021": ["t1", "t1ce", "t2", "flair"],
}
BRATS_LABEL_MAPS = {
    "2023": {0: 0, 1: 1, 2: 2, 3: 3},           # NCR, ED, ET
    "2021": {0: 0, 1: 1, 2: 2, 4: 3},           # ET stored as 4
    "2024": {0: 0, 1: 1, 2: 2, 3: 3, 4: 0},     # NETC, SNFH, ET; resection cavity -> bg
}


def read_jsonl(path: str | None) -> dict:
    if not path:
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[str(rec["case_id"])] = rec
    return out


def read_split(path: str | None):
    if not path:
        return None
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def _remap(label: np.ndarray, mapping: dict) -> np.ndarray:
    out = np.zeros_like(label, dtype=np.int64)
    for src, dst in mapping.items():
        out[label == src] = dst
    return out


class BraTSDataset(Dataset):
    def __init__(self, root: str, naming: str = "2023", label_map: str | None = None,
                 case_ids: Sequence[str] | None = None, split_file: str | None = None,
                 size=(128, 128, 128), train: bool = False, augment: dict | None = None,
                 reports_jsonl: str | None = None, class_index: int = 0,
                 tumor_crop_prob: float = 0.5, cache: bool = False):
        self.root = root
        self.mods = BRATS_MODALITIES[naming]
        self.label_map = BRATS_LABEL_MAPS[label_map or naming]
        self.naming = naming
        ids = case_ids or read_split(split_file)
        if ids is None:
            ids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.ids = list(ids)
        self.size = tuple(size) if size is not None else None   # None = full volume
        self.train = train
        self.aug = Augment(**(augment or {})) if train else None
        self.reports = read_jsonl(reports_jsonl)
        self.class_index = class_index
        self.tumor_crop_prob = tumor_crop_prob
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.ids)

    def _path(self, cid, suffix):
        sep = "-" if self.naming == "2023" else "_"
        cands = glob.glob(os.path.join(self.root, cid, f"{cid}{sep}{suffix}.nii*"))
        if not cands:
            raise FileNotFoundError(f"{cid}: missing {suffix}")
        return cands[0]

    def _load(self, cid):
        import nibabel as nib

        if self._cache is not None and cid in self._cache:
            return self._cache[cid]
        img = np.stack([np.asarray(nib.load(self._path(cid, m)).dataobj, dtype=np.float32)
                        for m in self.mods])
        try:
            seg = np.asarray(nib.load(self._path(cid, "seg")).dataobj).astype(np.int64)
            seg = _remap(seg, self.label_map)
            has = True
        except FileNotFoundError:
            seg = np.zeros(img.shape[1:], dtype=np.int64)
            has = False
        item = (zscore(torch.from_numpy(img)), torch.from_numpy(seg), has)
        if self._cache is not None:
            self._cache[cid] = item
        return item

    def __getitem__(self, i):
        cid = self.ids[i]
        img, seg, has = self._load(cid)
        bbox = foreground_bbox(img)
        center = None
        if self.train:
            tumor = (seg > 0).nonzero()
            if has and len(tumor) and random.random() < self.tumor_crop_prob:
                center = tumor[random.randrange(len(tumor))].tolist()
            else:
                center = [random.randint(lo, max(hi - 1, lo)) for lo, hi in bbox]
        if self.size is not None:
            img, seg = crop_or_pad(img, seg, self.size, center=center, bbox=bbox)
        if self.aug is not None:
            img, seg = self.aug(img, seg)
        rec = self.reports.get(cid, {})
        return {"image": img, "seg": seg, "has_mask": has, "cls": self.class_index,
                "report": rec.get("report", ""), "notes": rec.get("notes", ""), "case_id": cid}


def _resize2d(img: torch.Tensor, size, mode="bilinear"):
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(img[None], size=size, mode=mode, **kw)[0]


class FigShareDataset(Dataset):
    """Cheng et al. brain tumor dataset (.mat v7.3, fields cjdata.image/label/tumorMask).

    FigShare labels: 1 meningioma, 2 glioma, 3 pituitary.
    """

    FIGSHARE_TO_NAME = {1: "meningioma", 2: "glioma", 3: "pituitary"}

    def __init__(self, root: str, class_names: Sequence[str], files: Sequence[str] | None = None,
                 split_file: str | None = None, size=(224, 224), train=False,
                 augment: dict | None = None, mask_label: int = 1):
        self.files = list(files or read_split(split_file) or
                          sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)))
        self.files = [f if os.path.isabs(f) else os.path.join(root, f) for f in self.files]
        self.class_names = list(class_names)
        self.size = tuple(size)
        self.aug = Augment(**(augment or {})) if train else None
        self.mask_label = mask_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import h5py

        with h5py.File(self.files[i], "r") as f:
            d = f["cjdata"]
            img = np.array(d["image"], dtype=np.float32).T
            lab = int(np.array(d["label"]).squeeze())
            mask = np.array(d["tumorMask"], dtype=np.uint8).T
        img = _resize2d(torch.from_numpy(img)[None], self.size)
        mask = _resize2d(torch.from_numpy(mask)[None].float(), self.size, "nearest")[0]
        seg = (mask > 0.5).long() * self.mask_label
        img = zscore(img)
        if self.aug is not None:
            img, seg = self.aug(img, seg)
        cls = self.class_names.index(self.FIGSHARE_TO_NAME[lab])
        return {"image": img, "seg": seg, "has_mask": True, "cls": cls, "report": "",
                "notes": "", "case_id": os.path.splitext(os.path.basename(self.files[i]))[0]}


class FolderDataset(Dataset):
    """root/<class folder>/<image>; ``class_map`` maps folder names to class
    indices (use -100 for tumor-positive images without a type, e.g. Br35H 'yes')."""

    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, root: str, class_map: dict, size=(224, 224), train=False,
                 augment: dict | None = None, mask_root: str | None = None,
                 mask_label: int = 1, files: Sequence[str] | None = None):
        self.items = []
        if files is None:
            for folder, idx in class_map.items():
                for p in sorted(glob.glob(os.path.join(root, folder, "*"))):
                    if p.lower().endswith(self.EXTS):
                        self.items.append((p, int(idx), folder))
        else:
            for p in files:
                folder = os.path.basename(os.path.dirname(p))
                self.items.append((p if os.path.isabs(p) else os.path.join(root, p),
                                   int(class_map[folder]), folder))
        self.size = tuple(size)
        self.aug = Augment(**(augment or {})) if train else None
        self.mask_root = mask_root
        self.mask_label = mask_label

    def __len__(self):
        return len(self.items)

    def _mask(self, path, folder):
        if not self.mask_root:
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        for ext in self.EXTS:
            p = os.path.join(self.mask_root, folder, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, i):
        from PIL import Image

        path, cls, folder = self.items[i]
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        img = zscore(_resize2d(torch.from_numpy(img)[None], self.size))
        mpath = self._mask(path, folder)
        if mpath:
            m = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32)
            m = _resize2d(torch.from_numpy(m)[None], self.size, "nearest")[0]
            seg, has = (m > 127).long() * self.mask_label, True
        else:
            seg, has = torch.zeros(self.size, dtype=torch.long), False
        if self.aug is not None:
            img, seg = self.aug(img, seg)
        return {"image": img, "seg": seg, "has_mask": has, "cls": cls, "report": "",
                "notes": "", "case_id": os.path.splitext(os.path.basename(path))[0]}


def random_split_indices(n, test_fraction, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_fraction))
    return idx[n_test:], idx[:n_test]


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "has_mask": torch.tensor([bool(b["has_mask"]) for b in batch]),
        "cls": torch.tensor([int(b["cls"]) for b in batch]),
        "report": [b["report"] for b in batch],
        "notes": [b["notes"] for b in batch],
        "case_id": [b["case_id"] for b in batch],
    }
