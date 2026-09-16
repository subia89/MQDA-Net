import torch

from helpers import tiny_config, toy_report_generator
from mqda.losses.total import MQDALoss
from mqda.models.mqda_net import MQDANet
from mqda.utils.checkpoint import load_checkpoint, save_checkpoint
from mqda.utils.inference import sliding_window_segment


def _batch():
    x = torch.randn(2, 4, 32, 32, 32)
    y = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    y[:, 8:20, 10:22, 8:18] = 2
    y[:, 11:17, 13:19, 10:15] = 3
    y[:, 13:15, 15:17, 11:13] = 1
    return x, y


def test_vision_only_2d():
    torch.manual_seed(0)
    m = MQDANet(tiny_config(spatial_dims=2, in_channels=1))
    x = torch.randn(3, 1, 64, 64)
    y = torch.zeros(3, 64, 64, dtype=torch.long)
    y[:, 20:40, 20:40] = 1
    out = m(x, seg_labels=y)
    loss, logs = MQDALoss()(out, y, torch.tensor([0, 3, -100]), torch.tensor([True, True, False]))
    loss.backward()
    assert out["cls_logits"].shape == (3, 4) and float(logs["loss_txt"]) == 0.0


def test_full_pipeline_gradients_and_generation(tmp_path):
    torch.manual_seed(0)
    m = MQDANet(tiny_config(graph_from_ground_truth=True), toy_report_generator())
    x, y = _batch()
    out = m(x, seg_labels=y, reports=["Findings: ET 1 cm3.", ""], notes=["seizure", ""])
    assert "Measurements:" in out["prompts"][0]
    loss, logs = MQDALoss()(out, y, torch.tensor([0, 1]))
    assert float(logs["loss_txt"]) > 0 and float(logs["loss_align"]) >= 0
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], missing
    # frozen LLM backbone, trainable LoRA
    lora = [n for n, p in m.report.named_parameters() if p.requires_grad and "lora_" in n]
    frozen = [n for n, p in m.report.named_parameters() if not p.requires_grad]
    assert lora and frozen

    m.eval()
    gen = m.generate_reports(x, max_new_tokens=5)
    assert len(gen["reports"]) == 2

    path = tmp_path / "ck.pt"
    save_checkpoint(str(path), m, config={"a": 1})
    sd = torch.load(path, weights_only=False)["model"]
    assert not any(k.startswith("report.llm.") and "lora_" not in k for k in sd)
    m2 = MQDANet(tiny_config(), toy_report_generator())
    load_checkpoint(str(path), m2)
    for k, v in m.state_dict().items():
        if "lora_" in k or not k.startswith("report.llm."):
            assert torch.equal(v, m2.state_dict()[k]), k


def test_sliding_window_matches_input_size():
    m = MQDANet(tiny_config(use_graph=False)).eval()
    probs = sliding_window_segment(m, torch.randn(1, 4, 40, 36, 30), (32, 32, 32))
    assert probs.shape == (1, 4, 40, 36, 30)
    assert torch.allclose(probs.sum(1), torch.ones(1, 40, 36, 30), atol=1e-4)
