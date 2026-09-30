"""The two-stage protocol of Sec. 3.7 / Table 2: the deterministic 2D->3D input
adaptation, the classification step through the shared 3D encoder, and the
frozen vision modules of the report stage."""
import torch

from helpers import tiny_config, toy_report_generator
from mqda.data.adapt import adapt_2d_to_3d
from mqda.engine import freeze_vision_modules, set_train_mode
from mqda.losses.total import MQDALoss
from mqda.models.mqda_net import MQDANet


def test_adapt_2d_to_3d_shapes_and_content():
    x = torch.rand(3, 1, 224, 224)
    v = adapt_2d_to_3d(x, size=128, depth=16, channels=4)
    assert v.shape == (3, 4, 128, 128, 16)
    # every depth slice and every sequence channel is the same adapted image
    assert torch.allclose(v[:, 0, :, :, 0], v[:, 3, :, :, 15])
    assert v.requires_grad is False
    assert torch.allclose(adapt_2d_to_3d(x), adapt_2d_to_3d(x))     # deterministic


def test_classification_step_uses_shared_encoder_without_decoder():
    torch.manual_seed(0)
    m = MQDANet(tiny_config(adapt_size=32, adapt_depth=8))
    out = m(torch.rand(2, 1, 64, 64), task="cls")
    assert out["volumes"].shape == (2, 4, 32, 32, 8)
    assert out["cls_logits"].shape == (2, 4)
    assert "semantic_logits" not in out                     # decoder not in the path
    loss, logs = MQDALoss()(out, None, torch.tensor([0, 2]))
    loss.backward()
    enc = [p for p in m.encoder.parameters() if p.grad is not None]
    assert enc, "the classification step must update the shared encoder"
    assert all(p.grad is None for p in m.decoder.parameters())
    assert float(logs["loss_seg"]) == 0.0 and float(logs["loss_cls"]) > 0


def test_segmentation_step_skips_the_quantum_head():
    m = MQDANet(tiny_config())
    out = m(torch.randn(2, 4, 32, 32, 32), task="seg")
    assert "cls_logits" not in out
    y = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    y[:, 8:20, 8:20, 8:20] = 2
    loss, _ = MQDALoss()(out, y, None, torch.tensor([True, True]))
    loss.backward()
    assert all(p.grad is None for p in m.quantum_head.circuit.parameters())


def test_report_stage_freezes_the_vision_modules():
    m = MQDANet(tiny_config(), toy_report_generator())
    freeze_vision_modules(m)
    for mod in (m.encoder, m.decoder, m.quantum_head):
        assert all(not p.requires_grad for p in mod.parameters())
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("graph.") for n in trainable)
    assert any(n.startswith("qformer.") for n in trainable)
    assert any("lora_" in n for n in trainable)
    # the frozen modules stay in eval mode when training resumes (Table 2: evaluated, not updated)
    set_train_mode(m)
    assert m.training and m.graph.training and m.qformer.training
    assert not m.encoder.training and not m.decoder.training and not m.quantum_head.training
