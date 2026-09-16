import pytest
import torch

from mqda.losses.segmentation import BoundaryLoss, DiceCELoss, SegmentationLoss, one_hot
from mqda.models.decoder import DFCAM, DualBranchDecoder
from mqda.models.encoder import MSASPP, MQDAEncoder


@pytest.mark.parametrize("sd,shape", [(3, (2, 4, 32, 32, 32)), (2, (2, 1, 64, 64))])
def test_encoder_decoder_shapes(sd, shape):
    enc = MQDAEncoder(sd, shape[1], 8, (1, 1, 1, 1, 1), (2, 2, 2, 2, 2))
    dec = DualBranchDecoder(sd, enc.channels, 4, exp_r=2, dfcam_dim=16, dfcam_patch_sizes=(4, 8))
    feats = enc(torch.randn(shape))
    assert len(feats) == 5
    assert feats[-1].shape[2:] == tuple(s // 16 for s in shape[2:])
    out = dec(feats)
    assert out["binary_logits"].shape == (shape[0], 1, *shape[2:])
    assert out["semantic_logits"].shape == (shape[0], 4, *shape[2:])


def test_msaspp_preserves_shape():
    m = MSASPP(3, 40, (1, 2, 4, 6))
    x = torch.randn(1, 40, 6, 6, 6)
    assert m(x).shape == x.shape


def test_dfcam_is_residual_and_uses_binary_branch():
    m = DFCAM(3, 8, dim=16, patch_size=4)
    fs, fb = torch.randn(1, 8, 16, 16, 16), torch.randn(1, 8, 16, 16, 16)
    out1 = m(fs, fb)
    out2 = m(fs, fb + 1.0)
    assert out1.shape == fs.shape
    assert not torch.allclose(out1, out2)  # keys / values come from the binary features


def test_dice_ce_perfect_prediction_is_small():
    y = torch.randint(0, 4, (2, 16, 16))
    good = one_hot(y, 4) * 20.0
    bad = torch.zeros_like(good)
    loss = DiceCELoss()
    assert loss(good, y) < 0.05 < loss(bad, y)


def test_boundary_loss_zero_for_perfect():
    y = torch.zeros(1, 20, 20, dtype=torch.long)
    y[0, 5:15, 5:15] = 1
    y[0, 8:12, 8:12] = 3
    logits = one_hot(y, 4) * 30.0
    assert BoundaryLoss()(logits, y) < 1e-3


def test_segmentation_loss_skips_cases_without_masks():
    y = torch.randint(0, 4, (2, 8, 8))
    b, s = torch.randn(2, 1, 8, 8), torch.randn(2, 4, 8, 8)
    loss, _ = SegmentationLoss()(b, s, y, has_mask=torch.tensor([False, False]))
    assert float(loss) == 0.0
    loss, parts = SegmentationLoss()(b, s, y, has_mask=torch.tensor([True, False]))
    assert float(loss) > 0 and "seg_bce" in parts
