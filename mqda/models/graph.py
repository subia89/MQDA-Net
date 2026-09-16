"""Dynamic image-grounded knowledge graph + z-conditioned GAT (paper Sec. 3.5).

Per case, a graph G = (V, E) is assembled from the segmentation output:

Nodes
    * one patient node;
    * one sub-region node per detected sub-region (NCR / ED / ET), with
      attributes from the masked radiomic vector and pooled encoder features;
    * location nodes from atlas intersections of the tumor mask;
    * clinical-attribute (concept) nodes from the knowledge base;
    * symptom nodes extracted from clinical notes (when available).

Edges
    * anatomical: patient-sub-region, sub-region-location, atlas adjacency;
    * ontology: knowledge-base relations (e.g. ET -> abnormal enhancement);
    * learned: an edge classifier conditioned on the quantum measurement
      vector z scores candidate (node, concept) relations; its probabilities
      become soft edge weights, so the topology depends on z (Eq. 13).

A three-layer, four-head Graph Attention Network refines the nodes with
attention logits e_gh = LeakyReLU(a^T [W h_g || W h_h || W_m z]) (Eqs. 13-14)
and the graph embedding is g = MeanPool(h) (+) MaxPool(h).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import REGION_NAMES
from .kb import KnowledgeBase

NODE_PATIENT, NODE_REGION, NODE_LOCATION, NODE_CONCEPT, NODE_SYMPTOM = range(5)


@dataclass
class GraphBatch:
    node_type: torch.Tensor      # (B, N) long
    node_ref: torch.Tensor       # (B, N) long: region idx / atlas label / concept / symptom id
    node_scalar: torch.Tensor    # (B, N, 2): location overlap and hemisphere
    node_mask: torch.Tensor      # (B, N) bool
    fixed_adj: torch.Tensor      # (B, N, N) float in {0, 1}
    cand_mask: torch.Tensor      # (B, N, N) bool - candidate learned edges
    names: list                  # per case list of node names (for inspection)


class GraphBuilder:
    """Turns per-case measurements into padded graph structure tensors."""

    def __init__(self, kb: KnowledgeBase, atlas, max_locations: int = 8):
        self.kb = kb
        self.atlas = atlas
        self.max_locations = max_locations
        self.atlas_adj = atlas.adjacency() if atlas is not None else set()

    def build(self, measurements: list, notes: list | None = None, device="cpu") -> GraphBatch:
        kb = self.kb
        cases = []
        for b, m in enumerate(measurements):
            types, refs, scal, names = [NODE_PATIENT], [0], [(0.0, 0.0)], ["patient"]
            edges, cands = set(), set()
            region_nodes = {}
            for r, rn in enumerate(REGION_NAMES):
                if m.present.get(rn, False):
                    region_nodes[rn] = len(types)
                    types.append(NODE_REGION); refs.append(r); scal.append((0.0, 0.0)); names.append(rn)
                    edges.add((0, region_nodes[rn]))
            loc_nodes = {}
            for lid, lname, frac, hemi in m.locations[: self.max_locations]:
                loc_nodes[lid] = len(types)
                types.append(NODE_LOCATION); refs.append(lid); scal.append((frac, float(hemi)))
                names.append(lname)
                for rn, ri in region_nodes.items():
                    edges.add((ri, loc_nodes[lid]))
            for (a, c) in self.atlas_adj:
                if a in loc_nodes and c in loc_nodes:
                    edges.add((loc_nodes[a], loc_nodes[c]))
            concept_nodes = {}
            for ci, cname in enumerate(kb.concepts):
                concept_nodes[cname] = len(types)
                types.append(NODE_CONCEPT); refs.append(ci); scal.append((0.0, 0.0)); names.append(cname)
                cands.add((0, concept_nodes[cname]))
                for ri in region_nodes.values():
                    cands.add((ri, concept_nodes[cname]))
                for li in loc_nodes.values():
                    cands.add((li, concept_nodes[cname]))
            for rn, ri in region_nodes.items():
                for cname in kb.region_concepts.get(rn, []):
                    edges.add((ri, concept_nodes[cname]))
            for lid, li in loc_nodes.items():
                for cname in kb.concepts_for_location(self.atlas.names[lid]):
                    if cname in concept_nodes:
                        edges.add((li, concept_nodes[cname]))
            symptoms = kb.extract_symptoms(notes[b] if notes else None)
            for s in symptoms:
                si = len(types)
                types.append(NODE_SYMPTOM); refs.append(kb.symptom_names.index(s))
                scal.append((0.0, 0.0)); names.append(s)
                edges.add((0, si))
                for lid, li in loc_nodes.items():
                    if kb.symptom_matches_location(s, self.atlas.names[lid]):
                        edges.add((si, li))
            cases.append((types, refs, scal, names, edges, cands))

        n_max = max(len(c[0]) for c in cases)
        bsz = len(cases)
        node_type = torch.zeros(bsz, n_max, dtype=torch.long)
        node_ref = torch.zeros(bsz, n_max, dtype=torch.long)
        node_scalar = torch.zeros(bsz, n_max, 2)
        node_mask = torch.zeros(bsz, n_max, dtype=torch.bool)
        fixed = torch.zeros(bsz, n_max, n_max)
        cand = torch.zeros(bsz, n_max, n_max, dtype=torch.bool)
        all_names = []
        for b, (types, refs, scal, names, edges, cands) in enumerate(cases):
            n = len(types)
            node_type[b, :n] = torch.tensor(types)
            node_ref[b, :n] = torch.tensor(refs)
            node_scalar[b, :n] = torch.tensor(scal)
            node_mask[b, :n] = True
            fixed[b, torch.arange(n), torch.arange(n)] = 1.0
            for i, j in edges:
                fixed[b, i, j] = fixed[b, j, i] = 1.0
            for i, j in cands:
                cand[b, i, j] = cand[b, j, i] = True
            all_names.append(names)
        return GraphBatch(node_type.to(device), node_ref.to(device), node_scalar.to(device),
                          node_mask.to(device), fixed.to(device), cand.to(device), all_names)


class ZConditionedGATLayer(nn.Module):
    def __init__(self, dim, z_dim, heads=4, concat=True, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.head_dim = dim // heads if concat else dim
        self.W = nn.Linear(dim, self.head_dim * heads, bias=False)
        self.W_m = nn.Linear(z_dim, self.head_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_z = nn.Parameter(torch.empty(heads, self.head_dim))
        for p in (self.a_src, self.a_dst, self.a_z):
            nn.init.xavier_uniform_(p)
        self.slope = negative_slope
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, z, adj, node_mask):
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.head_dim)            # (B, N, H, D)
        wz = self.W_m(z).view(b, 1, self.heads, self.head_dim)          # (B, 1, H, D)
        s = (wh * self.a_src).sum(-1)                                   # (B, N, H)
        d = (wh * self.a_dst).sum(-1)
        zt = (wz * self.a_z).sum(-1)                                    # (B, 1, H)
        e = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1) + zt.unsqueeze(1), self.slope)  # (B,N,N,H)
        valid = (adj > 0) & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        e = e + torch.log(adj.clamp_min(1e-6)).unsqueeze(-1)            # soft edge weights
        e = e.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2).nan_to_num(0.0)
        alpha = self.dropout(alpha)
        out = torch.einsum("bghk,bhkd->bgkd", alpha, wh)                # (B, N, H, D)
        if self.concat:
            return out.reshape(b, n, -1)
        return out.mean(2)


class KnowledgeGraphReasoner(nn.Module):
    def __init__(self, region_feat_dim: int, radiomic_dim: int, global_dim: int,
                 n_location_labels: int, kb: KnowledgeBase, z_dim=12, dim=256,
                 layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.kb = kb
        self.type_emb = nn.Embedding(5, dim)
        self.patient_proj = nn.Linear(global_dim, dim)
        self.region_proj = nn.Linear(region_feat_dim + radiomic_dim, dim)
        self.region_emb = nn.Embedding(len(REGION_NAMES), dim)
        self.location_emb = nn.Embedding(n_location_labels, dim)
        self.location_scalar = nn.Linear(2, dim)
        self.concept_emb = nn.Embedding(len(kb.concepts), dim)
        self.symptom_emb = nn.Embedding(max(len(kb.symptom_names), 1), dim)
        self.edge_z = nn.Linear(z_dim, dim)
        self.edge_classifier = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.layers = nn.ModuleList(
            [ZConditionedGATLayer(dim, z_dim, heads, concat=(i < layers - 1), dropout=dropout)
             for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dim = dim

    def embed_nodes(self, gb: GraphBatch, global_feat, region_feat, region_rad):
        b, n = gb.node_type.shape
        h = self.type_emb(gb.node_type)
        t = gb.node_type
        ref = gb.node_ref
        # patient
        h = h + (t == NODE_PATIENT).unsqueeze(-1) * self.patient_proj(global_feat).unsqueeze(1)
        # sub-regions
        ridx = ref.clamp(max=len(REGION_NAMES) - 1)
        gather = lambda x: torch.gather(x, 1, ridx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))  # noqa
        reg = self.region_proj(torch.cat([gather(region_feat), gather(region_rad)], -1)) \
            + self.region_emb(ridx)
        h = h + (t == NODE_REGION).unsqueeze(-1) * reg
        # locations
        lidx = ref.clamp(max=self.location_emb.num_embeddings - 1)
        loc = self.location_emb(lidx) + self.location_scalar(gb.node_scalar)
        h = h + (t == NODE_LOCATION).unsqueeze(-1) * loc
        # concepts / symptoms
        cidx = ref.clamp(max=self.concept_emb.num_embeddings - 1)
        h = h + (t == NODE_CONCEPT).unsqueeze(-1) * self.concept_emb(cidx)
        sidx = ref.clamp(max=self.symptom_emb.num_embeddings - 1)
        h = h + (t == NODE_SYMPTOM).unsqueeze(-1) * self.symptom_emb(sidx)
        return h * gb.node_mask.unsqueeze(-1)

    def learned_edges(self, h, z, gb: GraphBatch):
        b, n, d = h.shape
        zz = self.edge_z(z).view(b, 1, 1, d).expand(b, n, n, d)
        pair = torch.cat([h.unsqueeze(2).expand(b, n, n, d), h.unsqueeze(1).expand(b, n, n, d), zz], -1)
        logits = self.edge_classifier(pair).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        prob = torch.sigmoid(logits) * gb.cand_mask
        return prob

    def forward(self, gb: GraphBatch, z, global_feat, region_feat, region_rad):
        z = z.float()
        h = self.embed_nodes(gb, global_feat, region_feat, region_rad)
        edge_prob = self.learned_edges(h, z, gb)
        adj = torch.maximum(gb.fixed_adj, edge_prob)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.elu(layer(h, z, adj, gb.node_mask)))
        mask = gb.node_mask.unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        mx = h.masked_fill(~mask, float("-inf")).max(1).values
        g = torch.cat([mean, mx], -1)
        return {"nodes": h, "graph_embedding": g, "adjacency": adj, "edge_prob": edge_prob}
