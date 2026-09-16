"""Small neuro-oncology knowledge base used for ontology-driven graph edges.

The lists below are intentionally compact and editable: extend them (or load
your own JSON with :func:`load_kb`) to encode a richer ontology.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CONCEPTS = [
    "abnormal enhancement",
    "central necrosis",
    "perilesional edema",
    "mass effect",
    "dural tail",
    "extra-axial location",
    "sellar location",
    "infiltrative margin",
]

SYMPTOMS = {
    "headache": [r"headache", r"cephalgia"],
    "seizure": [r"seizure", r"epilep", r"convuls"],
    "visual disturbance": [r"visual", r"diplopia", r"hemianopi", r"blurred vision"],
    "hemiparesis": [r"hemiparesis", r"weakness", r"hemiplegi"],
    "aphasia": [r"aphasia", r"dysphasia", r"speech"],
    "cognitive change": [r"confusion", r"memory", r"cognitive", r"personality"],
    "nausea": [r"nausea", r"vomit"],
    "hormonal imbalance": [r"hormon", r"amenorrh", r"galactorr", r"acromegal", r"cushing"],
}


@dataclass
class KnowledgeBase:
    concepts: list = field(default_factory=lambda: list(CONCEPTS))
    symptoms: dict = field(default_factory=lambda: dict(SYMPTOMS))
    # sub-region -> concepts it implies
    region_concepts: dict = field(default_factory=lambda: {
        "ET": ["abnormal enhancement"],
        "NCR": ["central necrosis"],
        "ED": ["perilesional edema", "mass effect", "infiltrative margin"],
    })
    # keyword in an atlas label name -> concepts
    location_concepts: dict = field(default_factory=lambda: {
        "sellar": ["sellar location"],
        "pituitary": ["sellar location"],
        "meninge": ["extra-axial location", "dural tail"],
        "convexity": ["extra-axial location"],
    })
    # symptom -> keywords of atlas label names it relates to
    symptom_locations: dict = field(default_factory=lambda: {
        "seizure": ["temporal", "frontal"],
        "visual disturbance": ["occipital", "sellar"],
        "hemiparesis": ["frontal", "parietal", "deep"],
        "aphasia": ["temporal", "frontal"],
        "cognitive change": ["frontal"],
        "hormonal imbalance": ["sellar"],
        "nausea": ["posterior fossa"],
    })

    @property
    def symptom_names(self):
        return list(self.symptoms.keys())

    def extract_symptoms(self, notes: str | None) -> list[str]:
        if not notes:
            return []
        text = notes.lower()
        found = []
        for name, patterns in self.symptoms.items():
            if any(re.search(p, text) for p in patterns):
                found.append(name)
        return found

    def concepts_for_location(self, label_name: str) -> list[str]:
        name = label_name.lower()
        out = []
        for key, concepts in self.location_concepts.items():
            if key in name:
                out += concepts
        return out

    def symptom_matches_location(self, symptom: str, label_name: str) -> bool:
        name = label_name.lower()
        return any(k in name for k in self.symptom_locations.get(symptom, []))


def load_kb(path: str | None) -> KnowledgeBase:
    if not path:
        return KnowledgeBase()
    with open(path) as f:
        data = json.load(f)
    return KnowledgeBase(**data)
