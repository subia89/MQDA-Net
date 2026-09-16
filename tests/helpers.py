"""Tiny configurations and an offline toy language model for tests."""
from __future__ import annotations

from mqda.models.mqda_net import ModelConfig


def tiny_config(spatial_dims=3, in_channels=4, **kw) -> ModelConfig:
    cfg = ModelConfig(
        spatial_dims=spatial_dims, in_channels=in_channels, base_channels=8,
        enc_blocks=[1, 1, 1, 1, 1], enc_exp=[2, 2, 2, 2, 2], dec_blocks=[1, 1, 1, 1],
        dec_exp=[2, 2, 2, 2], dfcam_dim=32, dfcam_patch_sizes=[4, 8], n_qubits=8,
        n_pos_qubits=2, n_kernels=4, q_hidden=[32], q_diff_method="backprop", gat_dim=32,
        gat_heads=4, qformer_layers=1, qformer_hidden=48, qformer_encoder_hidden=48,
        n_vis_queries=4, n_graph_queries=2, n_quant_queries=1, min_region_voxels=1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def toy_report_generator(hidden=64):
    from mqda.models.report import LLMConfig, ReportGenerator
    from mqda.models.toy_llm import toy_llm

    llm, tok = toy_llm(hidden)
    lcfg = LLMConfig(name_or_path="toy", lora_rank=4, lora_alpha=8, dtype="float32",
                     gradient_checkpointing=False, max_prompt_tokens=512,
                     max_report_tokens=128, align_dim=16)
    return ReportGenerator(lcfg, llm, tok)
