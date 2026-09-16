"""YAML config loading with ``key.sub=value`` command-line overrides."""
from __future__ import annotations

import copy
import dataclasses
import os

import yaml


def _set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        cfg = _merge(parent, cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        _set(cfg, key, yaml.safe_load(raw))
    return cfg


def to_dataclass(cls, data: dict | None):
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**data)
