# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Validated loading of SmellNet's class-level GC-MS fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Exact mapping released in MIT-MI/SmellNet/gcms_analysis/npz_to_csv.py.
_PRETTY_TO_SENSOR_LABEL = {
    "Peanut": "peanuts",
    "Cashew nut": "cashew",
    "Chestnut": "chestnuts",
    "Pistachio": "pistachios",
    "Almond": "almond",
    "Hazelnut": "hazelnut",
    "Common walnut": "walnuts",
    "Pecan nut": "pecans",
    "Brazil nut": "brazil nut",
    "Pili nut": "pili nut",
    "Cumin": "cumin",
    "Star anise": "star anise",
    "Nutmeg": "nutmeg",
    "Cloves": "cloves",
    "Ginger": "ginger",
    "Allspice": "allspice",
    "Chervil": "chervil",
    "White mustard": "mustard",
    "Cinnamon": "cinnamon",
    "Saffron": "saffron",
    "Angelica": "angelica",
    "Garlic": "garlic",
    "Chives": "chives",
    "Turnip": "turnip",
    "Dill": "dill",
    "Mugwort": "mugwort",
    "Roman camomile": "chamomile",
    "Coriander": "coriander",
    "Mexican oregano": "oregano",
    "Spearmint": "mint",
    "Kiwi": "kiwi",
    "Pineapple": "pineapple",
    "Banana": "banana",
    "Lemon": "lemon",
    "Mandarin orange (Clementine, Tangerine)": "mandarin orange",
    "Strawberry": "strawberry",
    "Apple": "apple",
    "Mango": "mango",
    "Peach": "peach",
    "Pear": "pear",
    "Cauliflower": "cauliflower",
    "Brussel sprouts": "brussel sprouts",
    "Broccoli": "broccoli",
    "Sweet potato": "sweet potato",
    "Asparagus": "asparagus",
    "Avocado": "avocado",
    "Radish": "radish",
    "Garden tomato": "tomato",
    "Potato": "potato",
    "Common cabbage": "cabbage",
}


@dataclass(frozen=True)
class SmellNetGCMSBank:
    labels: tuple[str, ...]
    features: torch.Tensor


def _canonical_label(value: object) -> str:
    return " ".join(str(value).replace("_", " ").split()).casefold()


def load_smellnet_gcms(
    path: str | Path,
    expected_labels: tuple[str, ...],
) -> SmellNetGCMSBank:
    """Load, feature-standardize, validate, and reorder the released 460-D bank."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SmellNet GC-MS asset not found: {path}")

    with np.load(path, allow_pickle=True) as payload:
        required = {"food_labels", "vectors"}
        missing_keys = sorted(required - set(payload.files))
        if missing_keys:
            raise ValueError(f"GC-MS asset is missing keys {missing_keys}: {path}")
        pretty_labels = [str(value) for value in payload["food_labels"].tolist()]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(pretty_labels):
        raise ValueError(f"invalid GC-MS shapes: labels={len(pretty_labels)} vectors={vectors.shape}")
    if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("GC-MS vectors must be non-empty and finite")

    unknown_pretty = sorted(set(pretty_labels) - set(_PRETTY_TO_SENSOR_LABEL))
    if unknown_pretty:
        raise ValueError(f"GC-MS asset contains unmapped food labels: {unknown_pretty}")
    labels = tuple(_canonical_label(_PRETTY_TO_SENSOR_LABEL[name]) for name in pretty_labels)
    if len(set(labels)) != len(labels):
        raise ValueError("GC-MS label mapping is not one-to-one")

    expected = tuple(_canonical_label(label) for label in expected_labels)
    if len(set(expected)) != len(expected):
        raise ValueError("expected SmellNet label vocabulary contains duplicates")
    missing = sorted(set(expected) - set(labels))
    unexpected = sorted(set(labels) - set(expected))
    if missing or unexpected:
        raise ValueError(f"GC-MS and SmellNet base vocabularies differ: missing={missing} unexpected={unexpected}")

    row_by_label = {label: vectors[index] for index, label in enumerate(labels)}
    ordered = np.stack([row_by_label[label] for label in expected]).astype(np.float32)

    # Match upstream models/load_data.py: StandardScaler over the class rows for
    # every m/z bin. Constant bins remain zero instead of producing NaN/Inf.
    mean = ordered.mean(axis=0, keepdims=True)
    scale = ordered.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (ordered - mean) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("standardized GC-MS features contain non-finite values")

    return SmellNetGCMSBank(
        labels=expected,
        features=torch.from_numpy(np.ascontiguousarray(standardized)),
    )
