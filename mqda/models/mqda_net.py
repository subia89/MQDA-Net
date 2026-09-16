"""MQDA-Net: unified segmentation, classification and report generation.

(Y_seg, y_cls, T_txt) = J_xi( E_psi( B_phi(C_theta(V)), O_omega(B_phi(C_theta(V))) ) )   (Eq. 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..data.atlas import build_atlas
from .decoder import DualBranchDecoder
from .encoder import MQDAEncoder
from .features import (REGION_NAMES, masked_average_pool, measure_case, radiomics,
                       region_probs, tumor_centroid)
from .graph import GraphBuilder, KnowledgeGraphReasoner
from .kb import load_kb
from .quantum import QuantumClassificationHead


@dataclass
class ModelConfig:
    spatial_dims: int = 3
    in_channels: int = 4
    num_seg_classes: int = 4
    class_names: list = field(default_factory=lambda: ["glioma", "meningioma", "pituitary", "no tumor"])
    modality_names: str = "T1, T1-CE, T2, FLAIR"
    voxel_volume_mm3: float = 1.0
    # encoder (MedNeXt-L style, block counts shortened at the bottleneck)
    base_channels: int = 32
    enc_blocks: list = field(default_factory=lambda: [3, 4, 8, 8, 2])
    enc_exp: list = field(default_factory=lambda: [3, 4, 8, 8, 8])
    kernel_size: int = 3
    aspp_rates: list = field(default_factory=lambda: [1, 2, 4, 6])
    # decoder
    dec_blocks: list = field(default_factory=lambda: [8, 8, 4, 3])
    dec_exp: list = field(default_factory=lambda: [8, 8, 4, 3])
    dfcam_stages: list = field(default_factory=lambda: [3, 4])
    dfcam_dim: int = 256
    dfcam_patch_sizes: list = field(default_factory=lambda: [8, 16])
    dfcam_heads: int = 1
    # quantum head
    n_qubits: int = 12
    n_pos_qubits: int = 3
    pqc_layers: int = 2
    n_kernels: int = 4
    q_hidden: list = field(default_factory=lambda: [2048, 1920])
    q_backend: str = "torch"
    q_diff_method: str = "parameter-shift"
    # knowledge graph
    use_graph: bool = True
    gat_dim: int = 256
    gat_layers: int = 3
    gat_heads: int = 4
    kb_path: Optional[str] = None
    atlas: dict = field(default_factory=lambda: {"type": "coarse"})
    min_region_voxels: int = 10
    graph_from_ground_truth: bool = False
    # Q-Former
    qformer_layers: int = 4
    qformer_hidden: int = 768
    qformer_encoder_hidden: int = 1408
    qformer_cross_freq: int = 1
    n_vis_queries: int = 32
    n_graph_queries: int = 8
    n_quant_queries: int = 4
    blip2_init: Optional[str] = None


class MQDANet(nn.Module):
    def __init__(self, cfg: ModelConfig, report_generator: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        sd = cfg.spatial_dims
        self.encoder = MQDAEncoder(sd, cfg.in_channels, cfg.base_channels, cfg.enc_blocks,
                                   cfg.enc_exp, cfg.kernel_size, cfg.aspp_rates)
        self.decoder = DualBranchDecoder(sd, self.encoder.channels, cfg.num_seg_classes,
                                         cfg.dec_exp, cfg.kernel_size, cfg.dec_blocks,
                                         cfg.dfcam_stages, cfg.dfcam_dim,
                                         cfg.dfcam_patch_sizes, cfg.dfcam_heads)
        c5 = self.encoder.bottleneck_channels
        self.rad_dim = 2 + 4 * cfg.in_channels
        n_reg = len(REGION_NAMES)
        self.cls_in_dim = n_reg * c5 + n_reg * self.rad_dim
        self.quantum_head = QuantumClassificationHead(
            self.cls_in_dim, len(cfg.class_names), cfg.n_qubits, cfg.n_pos_qubits,
            cfg.pqc_layers, cfg.n_kernels, cfg.q_hidden, backend=cfg.q_backend,
            diff_method=cfg.q_diff_method)

        self.kb = load_kb(cfg.kb_path)
        self.atlas = build_atlas(cfg.atlas)
        self.graph_builder = GraphBuilder(self.kb, self.atlas)
        self.graph = None
        if cfg.use_graph:
            self.graph = KnowledgeGraphReasoner(c5, self.rad_dim, c5, len(self.atlas.names),
                                                self.kb, cfg.n_qubits, cfg.gat_dim,
                                                cfg.gat_layers, cfg.gat_heads)
            self.graph_token = nn.Linear(2 * cfg.gat_dim, cfg.gat_dim)

        self.report = report_generator
        self.qformer = None
        if report_generator is not None:
            from .report import QFormerProjector

            self.qformer = QFormerProjector(
                c5, cfg.gat_dim, cfg.n_qubits, report_generator.hidden_size,
                cfg.n_vis_queries, cfg.n_graph_queries, cfg.n_quant_queries,
                cfg.qformer_layers, cfg.qformer_hidden, cfg.qformer_encoder_hidden,
                cfg.qformer_cross_freq, cfg.blip2_init)

    # ------------------------------------------------------------------
    def vision_parameters(self):
        mods = [self.encoder, self.decoder, self.quantum_head]
        if self.graph is not None:
            mods += [self.graph, self.graph_token]
        if self.qformer is not None:
            mods.append(self.qformer)
        for m in mods:
            yield from m.parameters()

    def language_parameters(self):
        if self.report is None:
            return iter(())
        return (p for p in self.report.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def segment_and_classify(self, images):
        feats = self.encoder(images)
        dec = self.decoder(feats)
        f5 = feats[-1]
        masks = region_probs(dec["semantic_logits"])                 # (N, 3, *S)
        pooled = masked_average_pool(f5, masks)                       # (N, 3, C5)
        rad = radiomics(images.float(), masks.float(), self.cfg.voxel_volume_mm3)
        f_cls = torch.cat([pooled.flatten(1), rad.flatten(1)], -1)
        tumor = masks.sum(1, keepdim=True)
        coords = tumor_centroid(tumor.float())
        logits, z = self.quantum_head(f_cls.float(), coords)
        return {"features": feats, "f5": f5, "binary_logits": dec["binary_logits"],
                "semantic_logits": dec["semantic_logits"], "region_masks": masks,
                "pooled": pooled, "radiomics": rad, "cls_logits": logits, "z": z,
                "coords": coords}

    @torch.no_grad()
    def measure(self, label_maps: torch.Tensor):
        is_3d = self.cfg.spatial_dims == 3
        ms = []
        for lab in label_maps:
            m = measure_case(lab, self.cfg.voxel_volume_mm3, self.cfg.min_region_voxels, is_3d)
            locs = self.atlas.locate(lab)
            m.locations = [(lid, name, frac, hemi) for lid, name, frac, hemi in locs]
            if m.centroid_vox:
                m.mni_mm = self.atlas.voxel_to_mni(m.centroid_vox, tuple(lab.shape))
            ms.append(m)
        return ms

    def build_graph(self, out, seg_labels=None, notes=None):
        if self.cfg.graph_from_ground_truth and seg_labels is not None and self.training:
            label_maps = seg_labels
        else:
            label_maps = out["semantic_logits"].argmax(1)
        measurements = self.measure(label_maps)
        gb = self.graph_builder.build(measurements, notes, device=out["f5"].device)
        f5 = out["f5"].float()
        global_feat = f5.flatten(2).mean(-1)
        g = self.graph(gb, out["z"], global_feat, out["pooled"].float(), out["radiomics"].float())
        return measurements, gb, g

    def prompts_for(self, out, measurements):
        probs = out["cls_logits"].float().softmax(-1)
        conf, idx = probs.max(-1)
        return [self.report.build_prompt(m.as_text(), self.cfg.class_names[i], c,
                                         self.cfg.modality_names)
                for m, i, c in zip(measurements, idx.tolist(), conf.tolist())]

    def condition_tokens(self, out, g):
        f5 = out["f5"]
        vis = f5.flatten(2).transpose(1, 2).float()                   # (N, T, C5)
        vis_mask = torch.ones(vis.shape[:2], dtype=torch.long, device=vis.device)
        gtok = torch.cat([g["nodes"], self.graph_token(g["graph_embedding"]).unsqueeze(1)], 1)
        gmask = torch.cat([g["node_mask"].long(),
                           torch.ones(gtok.shape[0], 1, dtype=torch.long, device=vis.device)], 1)
        return self.qformer(vis, vis_mask, gtok, gmask, out["z"])

    # ------------------------------------------------------------------
    def forward(self, images, seg_labels=None, reports=None, notes=None, with_report=None):
        out = self.segment_and_classify(images)
        with_report = (self.report is not None) if with_report is None else with_report
        if with_report and self.graph is not None and self.report is not None:
            measurements, gb, g = self.build_graph(out, seg_labels, notes)
            g["node_mask"] = gb.node_mask
            t_vis, t_graph, t_quant = self.condition_tokens(out, g)
            prompts = self.prompts_for(out, measurements)
            out.update(measurements=measurements, graph=g, graph_batch=gb, prompts=prompts,
                       tokens=(t_vis, t_graph, t_quant))
            if reports is not None:
                keep = [i for i, r in enumerate(reports) if r]
                if keep:
                    idx = torch.tensor(keep, device=images.device)
                    l_txt, l_align = self.report(t_vis[idx], t_graph[idx], t_quant[idx],
                                                 [prompts[i] for i in keep],
                                                 [reports[i] for i in keep])
                    out.update(loss_txt=l_txt, loss_align=l_align)
        return out

    @torch.no_grad()
    def generate_reports(self, images, notes=None, max_new_tokens=384, **kw):
        out = self.forward(images, notes=notes, with_report=True)
        t_vis, t_graph, t_quant = out["tokens"]
        texts = self.report.generate(t_vis, t_graph, t_quant, out["prompts"],
                                     max_new_tokens=max_new_tokens, **kw)
        out["reports"] = texts
        return out
