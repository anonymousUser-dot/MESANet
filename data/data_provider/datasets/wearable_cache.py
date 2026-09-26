"""Compact disk cache helpers for fixed-shape wearable forecasting chunks."""

from __future__ import annotations

import os
from pathlib import Path

import torch


_FORMAT = "wearable_stacked_chunks_v1"


def _pack_split(samples: list[dict]) -> dict:
    if not samples:
        return {"length": 0, "fields": {}, "kinds": {}}

    fields: dict[str, torch.Tensor] = {}
    kinds: dict[str, str] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, torch.Tensor):
            fields[key] = torch.stack(values, dim=0)
            kinds[key] = "tensor"
        elif isinstance(first, bool):
            fields[key] = torch.tensor(values, dtype=torch.bool)
            kinds[key] = "bool"
        elif isinstance(first, int):
            fields[key] = torch.tensor(values, dtype=torch.long)
            kinds[key] = "int"
        elif isinstance(first, float):
            fields[key] = torch.tensor(values, dtype=torch.float64)
            kinds[key] = "float"
        else:
            raise TypeError(f"unsupported wearable cache field {key!r}: {type(first)!r}")
    return {"length": len(samples), "fields": fields, "kinds": kinds}


def _unpack_split(packed: dict) -> list[dict]:
    samples: list[dict] = []
    fields = packed["fields"]
    kinds = packed["kinds"]
    for index in range(int(packed["length"])):
        sample: dict = {}
        for key, values in fields.items():
            kind = kinds[key]
            value = values[index]
            if kind == "tensor":
                sample[key] = value
            elif kind == "bool":
                sample[key] = bool(value.item())
            elif kind == "int":
                sample[key] = int(value.item())
            elif kind == "float":
                sample[key] = float(value.item())
            else:
                raise ValueError(f"unknown wearable cache kind {kind!r}")
        samples.append(sample)
    return samples


def stacked_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_stacked_v1{cache_path.suffix}")


def save_stacked_cache(cache_path: Path, splits: dict[str, list[dict]]) -> Path:
    """Persist equivalent chunks with one tensor storage per field and split."""

    target = stacked_cache_path(cache_path)
    payload = {
        "format": _FORMAT,
        "splits": {name: _pack_split(samples) for name, samples in splits.items()},
    }
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_wearable_splits(cache_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load a compact cache when available, otherwise create it from legacy data."""

    compact_path = stacked_cache_path(cache_path)
    if compact_path.exists():
        payload = torch.load(compact_path, map_location="cpu", weights_only=False)
        if payload.get("format") != _FORMAT:
            raise ValueError(f"unsupported wearable cache format in {compact_path}")
        splits = payload["splits"]
        return tuple(_unpack_split(splits[name]) for name in ("train", "val", "test"))

    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    splits = {name: cached[name] for name in ("train", "val", "test")}
    save_stacked_cache(cache_path, splits)
    return splits["train"], splits["val"], splits["test"]
