import torch

from mqda.data.atlas import CoarseLobeAtlas
from mqda.models.features import measure_case
from mqda.models.graph import (NODE_CONCEPT, NODE_LOCATION, NODE_REGION, NODE_SYMPTOM,
                               GraphBuilder, KnowledgeGraphReasoner)
from mqda.models.kb import KnowledgeBase


def _labels():
    y = torch.zeros(40, 40, 40, dtype=torch.long)
    y[26:36, 20:30, 20:30] = 2   # right hemisphere
    y[28:34, 22:28, 22:28] = 3
    y[30:32, 24:26, 24:26] = 1
    return y


def _measure(atlas):
    y = _labels()
    m = measure_case(y, 1.0, min_voxels=1)
    m.locations = atlas.locate(y)
    return m


def test_measurements():
    m = measure_case(_labels(), 1.0, min_voxels=1)
    assert abs(m.volumes_cm3["WT"] - 1.0) < 1e-9          # 10x10x10 voxels = 1 cm^3
    assert m.volumes_cm3["TC"] == (6 ** 3) / 1000
    assert abs(m.enhancement_ratio + m.necrosis_ratio - 1) < 1e-9
    assert all(m.present.values())


def test_atlas_laterality():
    atlas = CoarseLobeAtlas()
    locs = atlas.locate(_labels())
    assert locs and all(h == 1 for *_, h in locs)          # tumour is in the right half
    assert abs(sum(f for _, _, f, _ in locs) - 1.0) < 1e-6 or len(locs) == 8


def test_graph_structure_and_symptoms():
    kb, atlas = KnowledgeBase(), CoarseLobeAtlas()
    gb = GraphBuilder(kb, atlas).build([_measure(atlas), _measure(atlas)],
                                       notes=["new onset seizure", None])
    types = gb.node_type[0][gb.node_mask[0]].tolist()
    assert types.count(NODE_REGION) == 3
    assert types.count(NODE_CONCEPT) == len(kb.concepts)
    assert NODE_LOCATION in types and types.count(NODE_SYMPTOM) == 1
    assert gb.node_type[1][gb.node_mask[1]].tolist().count(NODE_SYMPTOM) == 0
    names = gb.names[0]
    et, enh = names.index("ET"), names.index("abnormal enhancement")
    assert gb.fixed_adj[0, et, enh] == 1                   # ontology edge
    assert gb.cand_mask[0, 0, enh]                        # learned candidate edge


def test_reasoner_is_z_conditioned():
    torch.manual_seed(0)
    kb, atlas = KnowledgeBase(), CoarseLobeAtlas()
    gb = GraphBuilder(kb, atlas).build([_measure(atlas)])
    r = KnowledgeGraphReasoner(16, 6, 16, len(atlas.names), kb, z_dim=12, dim=32).eval()
    args = (torch.randn(1, 16), torch.randn(1, 3, 16), torch.randn(1, 3, 6))
    o1 = r(gb, torch.zeros(1, 12), *args)
    o2 = r(gb, torch.ones(1, 12), *args)
    assert o1["graph_embedding"].shape == (1, 64)
    assert not torch.allclose(o1["edge_prob"], o2["edge_prob"])
    assert not torch.allclose(o1["graph_embedding"], o2["graph_embedding"])
    # padded / non-candidate positions never receive learned edges
    assert (o1["edge_prob"][~gb.cand_mask] == 0).all()
