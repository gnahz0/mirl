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
    * If ``data_source`` is in ``cfg.data.ts_data_sources``     -> ``branch="ts"``  (z_ts_img path).
    * Otherwise                                                 -> ``branch="img"`` (z_img path).
    * If the sample has an image entry, we use the first image  (``media_kind="image"``).
    * Else if the sample has a video entry, we use the first video frame-tensor
      (``media_kind="video"``, decoded via ``verl/utils/dataset/vision_utils.process_video``).
    * Else                                                       -> dropped.

Audio inputs are still TODO(stage2). Stage 1 only handles visual modalities.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import signal
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Match scripts/diagnose_skipped_videos.py: torchcodec is the preferred backend
# (avoids decord hangs on some MP4s) and we want it on in *all* processes including
# DataLoader workers, so set it at import time before qwen_vl_utils is touched.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


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


class _VideoTimeout(Exception):
    pass


def _video_timeout_handler(signum, frame):  # noqa: ARG001
    raise _VideoTimeout()


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Redirect file descriptor 2 to /dev/null while inside this context.

    Catches *C-level* writes from torchcodec / ffmpeg (``Could not open input file...``)
    that Python-level logging silencing can't reach. Falls through silently if the
    fd dance isn't possible (some sandboxed environments).
    """
    try:
        old_fd = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(old_fd, 2)
        finally:
            os.close(devnull)
            os.close(old_fd)


def _load_video_entry(
    video_entry: dict,
    image_patch_size: int = 14,
    max_frames_override: Optional[int] = None,
    timeout_sec: int = 30,
    suppress_stderr: bool = True,
):
    """Returns ``(video_tensor [n_frames, 3, H, W], video_metadata)`` or ``None``.

    Defensive wrapper around ``verl/utils/dataset/vision_utils.process_video``:

    * Sets a SIGALRM timeout (default 30s) so a malformed MP4 cannot hang the
      DataLoader. Matches the pattern from ``scripts/diagnose_skipped_videos.py``.
    * Suppresses C-level stderr from torchcodec / ffmpeg by default so unreadable
      files don't flood the console (the dataset already drops them silently).
    * Falls back to a plain ``process_video`` call if SIGALRM isn't usable in this
      thread / OS.

    Returns ``None`` on any failure (timeout, missing file, decode error) so the
    caller can route the sample to ``branch="none"`` and the collator drops it.
    """
    from verl.utils.dataset.vision_utils import process_video

    old_handler = None
    alarm_set = False
    try:
        try:
            old_handler = signal.signal(signal.SIGALRM, _video_timeout_handler)
            signal.alarm(int(timeout_sec))
            alarm_set = True
        except (ValueError, OSError):
            # SIGALRM unavailable (non-main thread, Windows, etc.); continue without timeout.
            pass

        ctx = _suppress_fd_stderr() if suppress_stderr else contextlib.nullcontext()
        with ctx:
            return process_video(
                video_entry,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
                max_frames_override=max_frames_override,
            )
    except _VideoTimeout:
        logger.warning("video decode timed out after %ds: %s", timeout_sec, video_entry)
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("video decode failed (%s): %s", type(e).__name__, str(e)[:160])
        return None
    finally:
        if alarm_set:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass


class AlignmentDataset(Dataset):
    """Yields per-sample dicts whose media is loaded lazily on ``__getitem__``.

    Each entry has::

        {
            "branch":      "img" | "ts" | "none",
            "media_kind":  "image" | "video" | "none",
            "media":       PIL.Image  OR  (video_tensor, video_metadata)  OR  None,
            "text":        str,            # text fed to CLIP
            "data_source": str,
            "index":       int,
        }

    Samples with ``branch == "none"`` are dropped by the collator.
    """

    def __init__(
        self,
        data_files: str | list[str],
        ts_data_sources: list[str],
        text_for_clip: str = "ground_truth",
        max_samples: int = -1,
        balanced_sampling_key: Optional[str] = None,
        seed: Optional[int] = 42,
        enable_videos: bool = True,
        max_video_frames: Optional[int] = 8,
        image_patch_size: int = 14,
        video_load_timeout: int = 30,
        video_suppress_stderr: bool = True,
    ):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.ts_sources = set(ts_data_sources or [])
        self.text_for_clip_mode = text_for_clip
        self.seed = seed
        self.enable_videos = enable_videos
        self.max_video_frames = max_video_frames
        self.image_patch_size = image_patch_size
        self.video_load_timeout = int(video_load_timeout)
        self.video_suppress_stderr = bool(video_suppress_stderr)

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

        def _has_media(r: dict) -> bool:
            if r.get("images"):
                return True
            if self.enable_videos and r.get("videos"):
                return True
            return False

        rows = [r for r in rows if _has_media(r)]
        if not rows:
            raise RuntimeError(
                f"no image/video-bearing samples found in {data_files} "
                f"(enable_videos={self.enable_videos})"
            )

        if max_samples and max_samples > 0:
            rows = self._maybe_stratified_sample(rows, max_samples, balanced_sampling_key)

        self.rows = rows
        logger.info(
            "AlignmentDataset: %d rows from %d files (videos=%s, max_video_frames=%s)",
            len(self.rows), len(data_files), self.enable_videos, self.max_video_frames,
        )

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

    def _none_item(self, idx: int, ds: str) -> dict:
        return {
            "branch": "none", "media_kind": "none", "media": None,
            "text": "", "data_source": ds, "index": idx,
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        ds = sample.get("data_source", "")
        branch = "ts" if ds in self.ts_sources else "img"

        # 1. Prefer images (cheaper, deterministic).
        images_meta = sample.get("images") or []
        if images_meta:
            pil = _load_image_path_or_dict(images_meta[0])
            if pil is not None:
                return {
                    "branch": branch, "media_kind": "image", "media": pil,
                    "text": _text_for_clip(sample, self.text_for_clip_mode),
                    "data_source": ds, "index": idx,
                }

        # 2. Fall back to videos.
        # NOTE: smellnet samples are image-only in the existing JSONL, so the ts branch is
        # effectively image-only. Tactile / HB / CLIMB video subsets feed the img branch.
        videos_meta = sample.get("videos") or []
        if self.enable_videos and videos_meta:
            loaded = _load_video_entry(
                videos_meta[0],
                image_patch_size=self.image_patch_size,
                max_frames_override=self.max_video_frames,
                timeout_sec=self.video_load_timeout,
                suppress_stderr=self.video_suppress_stderr,
            )
            if loaded is not None:
                return {
                    "branch": branch, "media_kind": "video", "media": loaded,
                    "text": _text_for_clip(sample, self.text_for_clip_mode),
                    "data_source": ds, "index": idx,
                }

        return self._none_item(idx, ds)


def collate_alignment(batch: list[dict]) -> dict:
    """Split per-sample dicts into 4 buckets keyed by (branch, media_kind).

    Returned dict::

        {
          "img_image_pil":   [PIL.Image, ...],     "img_image_text":   [str, ...],
          "img_video":       [(tensor, meta), ...],"img_video_text":   [str, ...],
          "ts_image_pil":    [PIL.Image, ...],     "ts_image_text":    [str, ...],
          "ts_video":        [(tensor, meta), ...],"ts_video_text":    [str, ...],
          "img_meta":        [{data_source, index, kind}, ...]  # in (images, videos) order
          "ts_meta":         [...]
        }

    Note: text order within each branch is ``[image_texts..., video_texts...]`` so the
    trainer can concatenate per-branch embeddings in the same order.
    """
    out = {
        "img_image_pil": [], "img_image_text": [],
        "img_video":     [], "img_video_text": [],
        "ts_image_pil":  [], "ts_image_text":  [],
        "ts_video":      [], "ts_video_text":  [],
        "img_meta":      [], "ts_meta":        [],
    }
    # First pass: images.
    for item in batch:
        if item["branch"] not in ("img", "ts"):
            continue
        if item["media_kind"] != "image" or item["media"] is None:
            continue
        prefix = item["branch"]
        out[f"{prefix}_image_pil"].append(item["media"])
        out[f"{prefix}_image_text"].append(item["text"])
        out[f"{prefix}_meta"].append({
            "data_source": item["data_source"], "index": item["index"], "kind": "image",
        })
    # Second pass: videos (appended after images in the per-branch order).
    for item in batch:
        if item["branch"] not in ("img", "ts"):
            continue
        if item["media_kind"] != "video" or item["media"] is None:
            continue
        prefix = item["branch"]
        out[f"{prefix}_video"].append(item["media"])
        out[f"{prefix}_video_text"].append(item["text"])
        out[f"{prefix}_meta"].append({
            "data_source": item["data_source"], "index": item["index"], "kind": "video",
        })
    return out
