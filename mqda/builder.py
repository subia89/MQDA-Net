"""Build MQDA-Net from a config dict."""
from __future__ import annotations

from .models.mqda_net import ModelConfig, MQDANet
from .utils.config import to_dataclass


def build_model(cfg: dict, with_report: bool | None = None) -> MQDANet:
    mcfg = to_dataclass(ModelConfig, cfg.get("model"))
    stage = cfg.get("train", {}).get("stage", "vision")
    if with_report is None:
        with_report = stage == "joint"
    report = None
    if with_report:
        from .models.report import LLMConfig, ReportGenerator

        report = ReportGenerator(to_dataclass(LLMConfig, cfg.get("llm")))
    return MQDANet(mcfg, report)
