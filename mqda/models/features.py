"""Region features for the classification head and the knowledge graph.

* masked average pooling of F5' inside each predicted sub-region;
* differentiable radiomic descriptors (volume, surface area, first-order
  intensity statistics per modality) computed from soft sub-region masks;
* tumor centroid (for the spatial qubits);
* hard, report-ready measurements (volumes in cm^3, enhancement and necrosis
  ratios) from the argmax segmentation.

Semantic label convention (BraTS 2023+):
    0 background, 1 necrotic / non-enhancing core (NCR), 2 edema (ED),
    3 enhancing tumor (ET).  TC = NCR + ET, WT = NCR + ED + ET.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


REGION_NAMES = ("NCR", "ED", "ET")


def region_probs(semantic_logits: torch.Tensor) -> torch.Tensor:
    """(N, 4, *S) logits -> (N, 3, *S) soft masks for NCR, ED, ET."""
    return semantic_logits.softmax(1)[:, 1:4]


def downsample_mask(mask: torch.Tensor, size) -> torch.Tensor:
    sd = mask.dim() - 2
    pool = F.adaptive_avg_pool3d if sd == 3 else F.adaptive_avg_pool2d
    return pool(mask, size)


def masked_average_pool(feat: torch.Tensor, masks: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """feat (N, C, *s), masks (N, R, *S) -> (N, R, C)."""
    m = downsample_mask(masks, feat.shape[2:])
    num = torch.einsum("nc...,nr...->nrc", feat, m)
    den = m.flatten(2).sum(-1, keepdim=True) + eps
    return num / den


def _spatial_grad_mag(p: torch.Tensor) -> torch.Tensor:
    sd = p.dim() - 2
    mags = 0
    for d in range(sd):
        diff = torch.diff(p, dim=2 + d)
        pad = [0, 0] * sd
        pad[2 * (sd - 1 - d) + 1] = 1
        mags = mags + F.pad(diff, pad).abs()
    return mags


def radiomics(image: torch.Tensor, masks: torch.Tensor, voxel_volume_mm3: float = 1.0,
              eps: float = 1e-6) -> torch.Tensor:
    """Soft radiomic descriptors per region.

    image (N, K, *S), masks (N, R, *S) -> (N, R, 2 + 4K):
    [log1p volume (cm^3), log1p surface (cm^2), then per modality
     mean, std, skewness, kurtosis inside the region].
    """
    n, r = masks.shape[:2]
    vol_vox = masks.flatten(2).sum(-1)                                 # (N, R)
    volume = torch.log1p(vol_vox * voxel_volume_mm3 / 1000.0)
    face = voxel_volume_mm3 ** (2.0 / 3.0) if image.dim() == 5 else voxel_volume_mm3
    surface = torch.log1p(_spatial_grad_mag(masks).flatten(2).sum(-1) * face / 100.0)
    w = masks / (vol_vox.view(n, r, *[1] * (masks.dim() - 2)) + eps)  # normalised weights
    x = image.unsqueeze(1)                                             # (N, 1, K, *S)
    wx = w.unsqueeze(2)                                                # (N, R, 1, *S)
    dims = tuple(range(3, x.dim()))
    mean = (wx * x).sum(dims)                                          # (N, R, K)
    centered = x - mean.view(*mean.shape, *[1] * len(dims))
    var = (wx * centered ** 2).sum(dims)
    std = torch.sqrt(var + eps)
    skew = (wx * centered ** 3).sum(dims) / (std ** 3 + eps)
    kurt = (wx * centered ** 4).sum(dims) / (var ** 2 + eps)
    stats = torch.stack([mean, std, skew.clamp(-10, 10), torch.log1p(kurt.clamp(0, 1e4))], -1)
    return torch.cat([volume.unsqueeze(-1), surface.unsqueeze(-1), stats.flatten(2)], -1)


def tumor_centroid(tumor_prob: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """(N, 1, *S) -> (N, 3) centroid in [-1, 1]; zeros for the missing axis in 2D."""
    n = tumor_prob.shape[0]
    sd = tumor_prob.dim() - 2
    p = tumor_prob[:, 0]
    total = p.flatten(1).sum(-1) + eps
    coords = []
    for d in range(sd):
        size = p.shape[1 + d]
        axis = torch.linspace(-1, 1, size, device=p.device)
        shape = [1] * sd
        shape[d] = size
        coords.append((p * axis.view(1, *shape)).flatten(1).sum(-1) / total)
    while len(coords) < 3:
        coords.append(torch.zeros(n, device=p.device))
    return torch.stack(coords, -1)


@dataclass
class CaseMeasurements:
    """Hard measurements used by the graph builder and the report prompt."""

    volumes_cm3: dict = field(default_factory=dict)   # NCR, ED, ET, TC, WT
    enhancement_ratio: float = 0.0                    # ET / TC
    necrosis_ratio: float = 0.0                       # NCR / TC
    present: dict = field(default_factory=dict)       # region -> bool
    centroid_vox: tuple = ()                          # voxel index of WT centroid
    locations: list = field(default_factory=list)     # [(label_id, name, overlap_frac, hemisphere)]
    mni_mm: tuple | None = None
    unit: str = "cm³"

    def as_text(self) -> str:
        v = self.volumes_cm3
        parts = [f"ET {v.get('ET', 0):.1f} {self.unit}", f"TC {v.get('TC', 0):.1f} {self.unit}",
                 f"WT {v.get('WT', 0):.1f} {self.unit}", f"edema {v.get('ED', 0):.1f} {self.unit}",
                 f"enhancement ratio {self.enhancement_ratio:.2f}",
                 f"necrosis ratio {self.necrosis_ratio:.2f}"]
        if self.locations:
            locs = ", ".join(f"{name} ({frac:.0%})" for _, name, frac, _ in self.locations[:4])
            parts.append(f"atlas regions: {locs}")
        if self.mni_mm is not None:
            x, y, z = self.mni_mm
            parts.append(f"centroid MNI x={x:.0f}, y={y:.0f}, z={z:.0f}")
        return "; ".join(parts)


@torch.no_grad()
def measure_case(labels: torch.Tensor, voxel_volume_mm3: float = 1.0,
                 min_voxels: int = 10, is_3d: bool = True) -> CaseMeasurements:
    """labels: (*S) integer map for ONE case."""
    counts = {name: int((labels == i + 1).sum()) for i, name in enumerate(REGION_NAMES)}
    counts["TC"] = counts["NCR"] + counts["ET"]
    counts["WT"] = counts["TC"] + counts["ED"]
    scale = voxel_volume_mm3 / (1000.0 if is_3d else 100.0)   # cm^3 (3D) or cm^2 (2D)
    m = CaseMeasurements(unit="cm³" if is_3d else "cm²")
    m.volumes_cm3 = {k: c * scale for k, c in counts.items()}
    tc = max(counts["TC"], 1)
    m.enhancement_ratio = counts["ET"] / tc
    m.necrosis_ratio = counts["NCR"] / tc
    m.present = {k: counts[k] >= min_voxels for k in REGION_NAMES}
    fg = (labels > 0).nonzero().float()
    m.centroid_vox = tuple(fg.mean(0).tolist()) if len(fg) else ()
    return m
