"""Cluster paths live in config.json (or $MIRL_DATA_ROOT), not in code."""

from __future__ import annotations

import json
import os
from pathlib import Path

_CONFIG = Path(__file__).with_name("config.json")


def data_root() -> str:
    if os.environ.get("MIRL_DATA_ROOT"):
        return os.environ["MIRL_DATA_ROOT"].rstrip("/")
    if _CONFIG.is_file():
        return str(json.loads(_CONFIG.read_text())["cluster_data_root"]).rstrip("/")
    return "data"  # fallback: relative, for laptop-local experiments


DATA_ROOT = data_root()
