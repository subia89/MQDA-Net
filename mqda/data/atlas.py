"""Anatomical atlases for laterality / lobe localisation (paper Sec. 3.5-3.6).

``NiftiAtlas`` intersects the tumor mask with a label atlas in MNI space
(for example the Harvard-Oxford cortical/subcortical atlases shipped with
FSL, or any lobe atlas). The input volume must already be registered to the
atlas space - see ``scripts/register_to_mni.py``.

``CoarseLobeAtlas`` is a dependency-free fallback that partitions the volume
box into hemisphere x lobe compartments by coordinate rules. It is only an
approximation of true anatomy and is meant for smoke tests and for data
without a registered atlas.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

# approximate MNI152 brain bounding box (mm)
MNI_BOX = ((-90.0, 90.0), (-126.0, 90.0), (-72.0, 108.0))


class BaseAtlas:
    names: list[str]

    def label_map(self, shape) -> torch.Tensor:
        raise NotImplementedError

    def adjacency(self) -> set[tuple[int, int]]:
        return set()

    def hemisphere(self, label_id: int) -> int:
        """+1 right, -1 left, 0 midline/unknown."""
        name = self.names[label_id].lower()
        if name.startswith("right") or " right" in name:
            return 1
        if name.startswith("left") or " left" in name:
            return -1
        return 0

    def voxel_to_mni(self, idx, shape):
        """Map a voxel index to approximate MNI mm by box scaling."""
        out = []
        for d in range(3):
            frac = idx[d] / max(shape[d] - 1, 1) if d < min(len(idx), len(shape)) else 0.5
            lo, hi = MNI_BOX[d]
            out.append(lo + frac * (hi - lo))
        return tuple(out)

    @torch.no_grad()
    def locate(self, labels: torch.Tensor, top_k: int = 8, min_frac: float = 0.02):
        """labels: (*S) segmentation of one case.

        Returns [(label_id, name, overlap_fraction, hemisphere)] sorted by
        overlap, restricted to atlas regions containing >= min_frac of the tumor.
        """
        tumor = labels > 0
        total = int(tumor.sum())
        if total == 0:
            return []
        lm = self.label_map(tuple(labels.shape)).to(labels.device)
        ids = lm[tumor]
        ids = ids[ids > 0]
        if ids.numel() == 0:
            return []
        counts = torch.bincount(ids.long(), minlength=len(self.names))
        order = torch.argsort(counts, descending=True)
        out = []
        for lid in order[:top_k].tolist():
            frac = counts[lid].item() / total
            if frac < min_frac:
                break
            out.append((lid, self.names[lid], frac, self.hemisphere(lid)))
        return out


class CoarseLobeAtlas(BaseAtlas):
    """Rule-based hemisphere x lobe partition of the volume box.

    Axis convention (configurable): axis 0 = left->right, axis 1 =
    posterior->anterior, axis 2 = inferior->superior; set the ``flip_*`` flags
    if your arrays are stored differently.
    """

    LOBES = ["frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe",
             "deep / periventricular", "posterior fossa", "sellar / suprasellar region"]

    def __init__(self, flip_lr=False, flip_ap=False, flip_is=False):
        self.flips = (flip_lr, flip_ap, flip_is)
        self.names = ["background"]
        for side in ("left", "right"):
            for lobe in self.LOBES:
                self.names.append(f"{side} {lobe}")

    def _id(self, side, lobe):
        return 1 + (0 if side == "left" else len(self.LOBES)) + self.LOBES.index(lobe)

    @lru_cache(maxsize=8)
    def _label_map_np(self, shape):
        sd = len(shape)
        axes = [np.linspace(0, 1, s) for s in shape]
        for d in range(sd):
            if self.flips[d]:
                axes[d] = axes[d][::-1]
        grids = np.meshgrid(*axes, indexing="ij")
        x = grids[0]
        y = grids[1]
        z = grids[2] if sd == 3 else np.full_like(x, 0.6)
        lat = np.abs(x - 0.5) * 2                    # 0 midline, 1 lateral
        right = x >= 0.5
        lobe = np.full(x.shape, self.LOBES.index("parietal lobe"))
        lobe[y > 0.6] = self.LOBES.index("frontal lobe")
        lobe[y < 0.25] = self.LOBES.index("occipital lobe")
        lobe[(z < 0.5) & (lat > 0.45) & (y >= 0.25)] = self.LOBES.index("temporal lobe")
        lobe[(lat < 0.25) & (z >= 0.35) & (z < 0.65) & (y >= 0.25) & (y <= 0.6)] = \
            self.LOBES.index("deep / periventricular")
        lobe[(z < 0.3) & (y < 0.45)] = self.LOBES.index("posterior fossa")
        lobe[(z < 0.35) & (lat < 0.15) & (y >= 0.45) & (y <= 0.65)] = \
            self.LOBES.index("sellar / suprasellar region")
        side_offset = np.where(right, len(self.LOBES), 0)
        return (1 + side_offset + lobe).astype(np.int64)

    def label_map(self, shape) -> torch.Tensor:
        return torch.from_numpy(self._label_map_np(tuple(shape)))

    def adjacency(self):
        pairs = {("frontal lobe", "parietal lobe"), ("frontal lobe", "temporal lobe"),
                 ("parietal lobe", "temporal lobe"), ("parietal lobe", "occipital lobe"),
                 ("temporal lobe", "occipital lobe"), ("deep / periventricular", "frontal lobe"),
                 ("deep / periventricular", "parietal lobe"),
                 ("deep / periventricular", "temporal lobe"),
                 ("posterior fossa", "occipital lobe"), ("posterior fossa", "temporal lobe"),
                 ("sellar / suprasellar region", "frontal lobe"),
                 ("sellar / suprasellar region", "temporal lobe")}
        out = set()
        for side in ("left", "right"):
            for a, b in pairs:
                out.add((self._id(side, a), self._id(side, b)))
        for lobe in ("frontal lobe", "parietal lobe", "occipital lobe",
                     "deep / periventricular", "posterior fossa", "sellar / suprasellar region"):
            out.add((self._id("left", lobe), self._id("right", lobe)))
        return out


class NiftiAtlas(BaseAtlas):
    """Label atlas stored as NIfTI with a lookup table ("id name" per line)."""

    def __init__(self, label_path: str, lut_path: str, adjacency_dilation: int = 1):
        import nibabel as nib

        img = nib.load(label_path)
        self.data = np.asarray(img.dataobj).astype(np.int64)
        self.affine = img.affine
        max_id = int(self.data.max())
        self.names = [f"label {i}" for i in range(max_id + 1)]
        self.names[0] = "background"
        with open(lut_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                idx, name = line.split(maxsplit=1)
                if int(idx) <= max_id:
                    self.names[int(idx)] = name
        self._adj = self._compute_adjacency(adjacency_dilation)

    def _compute_adjacency(self, r):
        from scipy import ndimage

        adj = set()
        ids = [i for i in np.unique(self.data) if i > 0]
        for i in ids:
            grown = ndimage.binary_dilation(self.data == i, iterations=r)
            for j in np.unique(self.data[grown]):
                if j > 0 and j != i:
                    adj.add((int(min(i, j)), int(max(i, j))))
        return adj

    def adjacency(self):
        return self._adj

    @lru_cache(maxsize=8)
    def _resampled(self, shape):
        t = torch.from_numpy(self.data)[None, None].float()
        t = torch.nn.functional.interpolate(t, size=shape, mode="nearest")
        return t[0, 0].long()

    def label_map(self, shape):
        return self._resampled(tuple(shape))

    def voxel_to_mni(self, idx, shape):
        scale = [self.data.shape[d] / shape[d] for d in range(3)]
        vox = np.array([idx[d] * scale[d] for d in range(3)] + [1.0])
        return tuple((self.affine @ vox)[:3].tolist())


def build_atlas(cfg: dict | None) -> BaseAtlas:
    cfg = cfg or {}
    if cfg.get("type", "coarse") == "nifti":
        return NiftiAtlas(cfg["label_path"], cfg["lut_path"], cfg.get("adjacency_dilation", 1))
    return CoarseLobeAtlas(cfg.get("flip_lr", False), cfg.get("flip_ap", False),
                           cfg.get("flip_is", False))
