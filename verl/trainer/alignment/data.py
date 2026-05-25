# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Dataset wrapper for Stage 1 multimodal alignment.

Reuses the repo's existing JSONL schema (no schema changes):
    data_source: str
    prompt: [{role, content}]               (content has <image>/<video>/<audio> placeholders)
    images: [{"image": "/abs/path.png", ...}]
    videos: [{"video": "...mp4", "max_frames": int}]
    audios: [{"audio": "...wav"}]
    reward_model: {"style": "rule", "ground_truth": str}
    extra_info: dict or JSON string

Routing logic:
    * If ``data_source`` is in ``cfg.data.ts_data_sources`` AND ``images`` is non-empty,
      we treat the first image as a *rendered time-series image* (z_ts_img path).
    * Otherwise if ``images`` is non-empty, we treat the first image as a *normal image*
      (z_img path).
    * Otherwise (video-only / audio-only / text-only) the sample is currently skipped --
      see TODO(stage2) below for adding video frame sampling and audio support.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


_PLACEHOLDER_RE = re.compile(r"<image>|<video>|<audio>")


def _parse_extra_info(ei) -> dict:
    if ei is None:
        return {}
    if isinstance(ei, dict):
        return ei
    if isinstance(ei, str):
        try:
            return json.loads(ei)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text).strip()


def _user_question(prompt: list[dict]) -> str:
    """Last user-turn text content with <image>/<video>/<audio> placeholders removed."""
    for msg in reversed(prompt or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return _strip_placeholders(content)
            if isinstance(content, list):
                parts = []
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(it.get("text", ""))
                return _strip_placeholders(" ".join(parts))
    return ""


def _text_for_clip(sample: dict, mode: str) -> str:
    gt = (sample.get("reward_model") or {}).get("ground_truth", "")
    if isinstance(gt, list):
        gt = ", ".join(str(x) for x in gt)
    gt = str(gt or "").strip()
    if mode == "ground_truth":
        return gt
    if mode == "question":
        return _user_question(sample.get("prompt") or [])
    if mode == "question_plus_gt":
        q = _user_question(sample.get("prompt") or [])
        return f"{q} | {gt}".strip(" |")
    raise ValueError(f"unknown text_for_clip mode {mode!r}")


def _load_image_path_or_dict(img_entry) -> Optional[Image.Image]:
    """Lazy, robust image load. Returns ``None`` on any failure so the sample can be dropped."""
    try:
        from verl.utils.dataset.vision_utils import process_image
        return process_image(img_entry)
    except Exception:  # noqa: BLE001
        try:
            if isinstance(img_entry, Image.Image):
                return img_entry.convert("RGB")
            if isinstance(img_entry, dict):
                path = img_entry.get("image") or img_entry.get("path")
                if path and os.path.exists(path):
                    return Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    return None


class AlignmentDataset(Dataset):
    """Yields per-sample dicts with PIL images on the CPU; the trainer's collate runs the
    Qwen processor + CLIP tokenizer in batched mode for efficiency.

    Each ``__getitem__`` returns ::

        {
            "branch":     "img" | "ts" | "none",
            "image":      Optional[PIL.Image],         # the single chosen image
            "text":       str,                          # text fed to CLIP
            "data_source": str,
            "index":      int,
        }

    Samples with ``branch == "none"`` are filtered out at index-time so the DataLoader
    never sees them.
    """

    def __init__(
        self,
        data_files: str | list[str],
        ts_data_sources: list[str],
        text_for_clip: str = "ground_truth",
        max_samples: int = -1,
        balanced_sampling_key: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.ts_sources = set(ts_data_sources or [])
        self.text_for_clip_mode = text_for_clip
        self.seed = seed

        rows: list[dict] = []
        for path in data_files:
            with open(path, "r") as f:
                if path.endswith(".jsonl") or path.endswith(".json"):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    raise ValueError(f"unsupported data file: {path}")

        # Pre-filter rows that have at least one image (Stage 1 supports image-bearing samples only).
        # TODO(stage2): support video frame sampling and audio inputs here.
        rows = [r for r in rows if r.get("images")]
        if not rows:
            raise RuntimeError(f"no image-bearing samples found in {data_files}")

        if max_samples and max_samples > 0:
            rows = self._maybe_stratified_sample(rows, max_samples, balanced_sampling_key)

        self.rows = rows

    def _maybe_stratified_sample(
        self, rows: list[dict], n: int, key: Optional[str]
    ) -> list[dict]:
        rng = random.Random(self.seed)
        if not key:
            rng.shuffle(rows)
            return rows[:n]
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key, "__unknown__")), []).append(r)
        n_groups = len(buckets)
        per = max(1, n // n_groups)
        extra = n - per * n_groups
        out: list[dict] = []
        for i, (_, lst) in enumerate(sorted(buckets.items())):
            rng.shuffle(lst)
            take = per + (1 if i < extra else 0)
            out.extend(lst[:take])
        rng.shuffle(out)
        logger.info("stratified sample: %d from %d (%d groups by %r)",
                    len(out), sum(len(v) for v in buckets.values()), n_groups, key)
        return out[:n]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        ds = sample.get("data_source", "")
        images_meta = sample.get("images") or []
        if not images_meta:
            return {"branch": "none", "image": None, "text": "",
                    "data_source": ds, "index": idx}

        pil = _load_image_path_or_dict(images_meta[0])
        if pil is None:
            return {"branch": "none", "image": None, "text": "",
                    "data_source": ds, "index": idx}

        branch = "ts" if ds in self.ts_sources else "img"
        return {
            "branch": branch,
            "image": pil,
            "text": _text_for_clip(sample, self.text_for_clip_mode),
            "data_source": ds,
            "index": idx,
        }


def collate_alignment(batch: list[dict]) -> dict:
    """Group per-sample dicts into a single batch dict, keeping ``"img"`` and ``"ts"``
    branches separate so each can be processed independently by the Qwen processor."""
    img_pil, ts_pil = [], []
    img_text, ts_text = [], []
    img_meta, ts_meta = [], []
    for item in batch:
        if item["branch"] == "img" and item["image"] is not None:
            img_pil.append(item["image"]); img_text.append(item["text"])
            img_meta.append({"data_source": item["data_source"], "index": item["index"]})
        elif item["branch"] == "ts" and item["image"] is not None:
            ts_pil.append(item["image"]); ts_text.append(item["text"])
            ts_meta.append({"data_source": item["data_source"], "index": item["index"]})
        # branch == "none" is silently dropped
    return {
        "img_pil": img_pil, "img_text": img_text, "img_meta": img_meta,
        "ts_pil": ts_pil,   "ts_text": ts_text,   "ts_meta": ts_meta,
    }
