import numpy as np
import torch
import torch.nn.functional as F

from mqda.eval.metrics import dice_iou, hd95_assd, segmentation_metrics
from mqda.eval.stats import bonferroni, paired_test
from mqda.losses.classification import FocalLabelSmoothingLoss
from mqda.losses.preference import dpo_loss
from mqda.models.report import info_nce


def test_focal_reduces_to_cross_entropy():
    logits, y = torch.randn(8, 4), torch.randint(0, 4, (8,))
    loss = FocalLabelSmoothingLoss(gamma=0.0, beta=0.0)(logits, y)
    assert torch.allclose(loss, F.cross_entropy(logits, y), atol=1e-6)


def test_focal_matches_formula():
    logits, y = torch.randn(5, 4), torch.randint(0, 4, (5,))
    g, b = 2.0, 0.1
    p = logits.softmax(-1)
    alpha = torch.full_like(p, b / 3)
    alpha[torch.arange(5), y] = 1 - b
    manual = -(alpha * (1 - p) ** g * p.log()).sum(-1).mean()
    assert torch.allclose(FocalLabelSmoothingLoss(g, b)(logits, y), manual, atol=1e-6)


def test_focal_ignores_unknown_labels():
    logits = torch.randn(3, 4)
    y = torch.tensor([-100, -100, -100])
    assert float(FocalLabelSmoothingLoss()(logits, y)) == 0.0


def test_info_nce_prefers_matched_pairs():
    v = torch.eye(4)
    assert info_nce(v, v) < info_nce(v, v.flip(0))


def test_dpo_loss_sign():
    pc, pr = torch.tensor([0.0]), torch.tensor([-5.0])
    lo, _ = dpo_loss(pc, pr, torch.tensor([-1.0]), torch.tensor([-1.0]))
    hi, _ = dpo_loss(pr, pc, torch.tensor([-1.0]), torch.tensor([-1.0]))
    assert lo < hi


def test_dice_and_distances():
    a = np.zeros((20, 20, 20), bool)
    a[5:15, 5:15, 5:15] = True
    assert dice_iou(a, a) == (1.0, 1.0)
    b = np.roll(a, 2, axis=0)
    d, j = dice_iou(a, b)
    assert abs(d - 0.8) < 1e-6 and abs(j - 8 / 12) < 1e-6
    hd, assd = hd95_assd(a, b)
    assert 1.0 <= hd <= 2.0 + 1e-6 and 0 < assd <= 2.0
    assert hd95_assd(a, a) == (0.0, 0.0)


def test_region_metrics_keys():
    y = np.zeros((10, 10), int)
    y[2:8, 2:8] = 2
    y[3:6, 3:6] = 3
    m = segmentation_metrics(y, y)
    assert m["dice_mean"] == 1.0 and m["hd95_WT"] == 0.0


def test_paired_test_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.random(40)
    res = paired_test(base + 0.1 + rng.normal(0, 0.01, 40), base, n_resamples=2000)
    assert res["p_value"] < 0.001 and res["ci_low"] > 0.05 and res["rank_biserial"] > 0.9
    corr = bonferroni({"a": 0.001, "b": 0.04})
    assert corr["a"]["significant"] and not corr["b"]["significant"]
