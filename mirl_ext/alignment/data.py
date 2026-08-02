# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import re
import signal
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# Match scripts/diagnose_skipped_videos.py: torchcodec is the preferred backend
# (avoids decord hangs on some MP4s) and we want it on in *all* processes including
# DataLoader workers, so set it at import time before qwen_vl_utils is touched.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


_PLACEHOLDER_RE = re.compile(r"<image>|<video>|<audio>")
_HAPTIC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[_-]?")
_HAPTIC_INDEX_RE = re.compile(r"idx\d+$", re.IGNORECASE)

# 3DHaptic stems are ``date_[participant_]task[_detail...]_[participant_]idx``.
# The participant field moved between the prefix and suffix in early captures, and
# the 2026 captures use anonymous IDs. Keep the closed list here so object/task
# words can never be discarded merely because they happen to occupy one position.
_HAPTIC_PARTICIPANTS = frozenset(
    "aaa bbb ccc ddd eee amy arman arnie bonnie brandon cassius chaerin david "
    "devin han ian ivy jiayi julia koda leo luca mingxi mingxi2 paul rao runfeng "
    "sarah shaliz simin souju vivian xiaoyan yichen yinghua yubo zach ziyi".split()
)
_HAPTIC_HIERARCHICAL_TASKS = frozenset(
    {
        "actuation",
        "compliance",
        "deformable",
        "elastic",
        "inhand",
        "insert",
        "lift",
        "multiobject",
        "pour",
        "pull",
        "push",
        "recognition",
        "rubber",
        "slip",
        "teapot",
    }
)


def _canonical_label_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _haptic_stem_task_label(sample: dict) -> str:
    """Return the reusable two-level task encoded in a 3DHaptic clip stem."""
    stem = str(_mapping(sample.get("extra_info")).get("stem", "")).strip()
    if not stem:
        raise ValueError("tactile_label_mode='stem_task_pair' requires extra_info.stem")

    body = _HAPTIC_DATE_RE.sub("", stem.casefold())
    tokens = [token for token in re.split(r"[_-]+", body) if token]
    if tokens and _HAPTIC_INDEX_RE.fullmatch(tokens[-1]):
        tokens.pop()
    while tokens and tokens[0] in _HAPTIC_PARTICIPANTS:
        tokens.pop(0)
    while tokens and tokens[-1] in _HAPTIC_PARTICIPANTS:
        tokens.pop()
    if not tokens:
        # One legacy recording is named only by its participant. Keep its signal in
        # every objective instead of silently dropping it or preventing startup.
        return "unclassified haptic task"

    width = 2 if tokens[0] in _HAPTIC_HIERARCHICAL_TASKS and len(tokens) > 1 else 1
    return _canonical_label_text(" ".join(tokens[:width]))


def _user_question(prompt: list[dict]) -> str:
    """Last user-turn text content with <image>/<video>/<audio> placeholders removed."""
    for msg in reversed(prompt or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return _PLACEHOLDER_RE.sub(" ", content).strip()
            if isinstance(content, list):
                parts = []
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(it.get("text", ""))
                return _PLACEHOLDER_RE.sub(" ", " ".join(parts)).strip()
    return ""


def _text_for_label_encoder(
    sample: dict,
    mode: str,
    *,
    family: Optional[str] = None,
    tactile_label_mode: str = "ground_truth",
) -> str:
    if family == "tactile" and tactile_label_mode == "stem_task_pair":
        return _haptic_stem_task_label(sample)
    if tactile_label_mode not in {"ground_truth", "stem_task_pair"}:
        raise ValueError(f"unknown tactile_label_mode {tactile_label_mode!r}")

    gt = (sample.get("reward_model") or {}).get("ground_truth", "")
    if isinstance(gt, list):
        gt = ", ".join(str(x) for x in gt)
    gt = str(gt or "").strip()
    if mode == "ground_truth":
        text = gt
    elif mode == "question":
        text = _user_question(sample.get("prompt") or [])
    elif mode == "question_plus_gt":
        q = _user_question(sample.get("prompt") or [])
        text = f"{q} | {gt}".strip(" |")
    else:
        raise ValueError(f"unknown text_for_label mode {mode!r}")
    return _canonical_label_text(text)


def _load_image_path_or_dict(img_entry) -> Optional[Image.Image]:
    """Load one image, dropping unreadable rows."""
    try:
        from verl.utils.dataset.vision_utils import process_image

        return process_image(img_entry)
    except Exception:  # noqa: BLE001
        return None


_TIME_COL_HINTS = ("timestamp", "time_ms", "time")
_SIGNAL_FAMILIES = ("smell", "ecg", "tactile")


def _signal_path(sig_entry) -> Optional[str]:
    if isinstance(sig_entry, str):
        return sig_entry
    if isinstance(sig_entry, dict):
        return sig_entry.get("signal") or sig_entry.get("path") or sig_entry.get("csv")
    return None


def _signal_family(sig_entry) -> str:
    """Return the closed alignment family for one signal descriptor."""
    fmt = sig_entry.get("format") if isinstance(sig_entry, dict) else None
    if fmt == "ts_pt":
        return "ecg"
    if fmt == "tactile_pt":
        return "tactile"
    return "smell"


def _load_signal_csv(sig_entry) -> Optional[torch.Tensor]:
    """Load a raw CSV as native-shape ``(channels, time)``, dropping time columns."""
    path = _signal_path(sig_entry)
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip().split(",")
    keep_idx = [i for i, name in enumerate(header) if not any(h in name.strip().lower() for h in _TIME_COL_HINTS)]
    if not keep_idx:
        return None
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=keep_idx,
        dtype=np.float32,
    )
    if data.ndim == 1:
        data = data[:, None] if len(keep_idx) == 1 else data[None, :]
    arr = np.asarray(data, dtype=np.float32).T
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _load_signal_pt_1d(path: str) -> Optional[torch.Tensor]:
    """Load a multivariate ``(channels, time)`` tensor from ``.pt``."""
    if not path or not os.path.exists(path):
        return None
    t = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = next(t[key] for key in ("signal", "data", "ts") if t.get(key) is not None)
    t = torch.as_tensor(t).float()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t.contiguous()


def _load_tactile_pt(
    path: str,
    key: str = "glove_right",
    max_frames: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Load right tactile maps and aligned right-hand force statistics.

    v1 stores one right tactile map and 13 right-force columns. v2 additionally
    stores left/aligned/mat maps and 26 force columns; for a schema consistent with
    v1 and the requested ``glove_right`` signal, only the 13 ``right_*`` columns
    accompany the selected right map. One known v1 record has no force statistics;
    it receives an empty force tensor and remains usable.
    """
    if not path or not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    tac = (d.get("tactile") or {}).get(key) if isinstance(d, dict) else d
    if tac is None:
        return None
    t = torch.as_tensor(tac).float()
    if t.numel() == 0:
        return None
    if t.ndim == 2:
        t = t.unsqueeze(0)

    force = None
    columns = d.get("hand_force_columns") if isinstance(d, dict) else None
    force_raw = d.get("hand_force_stats") if isinstance(d, dict) else None
    if force_raw is not None and columns:
        force_all = torch.as_tensor(force_raw).float()
        right_idx = [i for i, name in enumerate(columns) if str(name).startswith("right_")]
        if right_idx:
            force = force_all[:, right_idx]
            if force.shape[0] > 1 and force.shape[0] != t.shape[0]:
                force = (
                    F.interpolate(
                        force.t().unsqueeze(0),
                        size=t.shape[0],
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .t()
                )
            elif force.shape[0] == 1:
                force = force.expand(t.shape[0], -1)
    if force is None:
        force = t.new_empty((t.shape[0], 0))

    if max_frames and t.shape[0] > max_frames:
        idx = torch.linspace(0, t.shape[0] - 1, int(max_frames)).round().long()
        t = t[idx]
        force = force[idx]
    return {"tactile": t.contiguous(), "force": force.contiguous()}


class _VideoTimeout(Exception):
    pass


def _video_timeout_handler(signum, frame):  # noqa: ARG001
    raise _VideoTimeout()


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Suppress C-level torchcodec/ffmpeg stderr."""
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(devnull)
        os.close(old_fd)


def _load_video_entry(
    video_entry: dict,
    image_patch_size: int = 14,
    max_frames_override: Optional[int] = None,
    timeout_sec: int = 30,
    suppress_stderr: bool = True,
):
    """Decode a video with a timeout, returning ``None`` on failure."""
    from verl.utils.dataset.vision_utils import process_video

    old_handler = signal.signal(signal.SIGALRM, _video_timeout_handler)
    signal.alarm(int(timeout_sec))
    try:
        ctx = _suppress_fd_stderr() if suppress_stderr else contextlib.nullcontext()
        with ctx:
            return process_video(
                video_entry,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
                nframes=max_frames_override,
            )
    except _VideoTimeout:
        logger.warning("video decode timed out after %ds: %s", timeout_sec, video_entry)
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("video decode failed (%s): %s", type(e).__name__, str(e)[:160])
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class AlignmentDataset(Dataset):
    """Yield lazily loaded image, video, or native-signal samples."""

    def __init__(
        self,
        data_files: str | list[str],
        text_for_label: str = "ground_truth",
        tactile_label_mode: str = "ground_truth",
        max_samples: int = -1,
        balanced_sampling_key: Optional[str] = None,
        seed: Optional[int] = 42,
        enable_videos: bool = True,
        max_video_frames: Optional[int] = 8,
        image_patch_size: int = 14,
        video_load_timeout: int = 30,
        video_suppress_stderr: bool = True,
        data_source_filter: Optional[list[str]] = None,
        exclude_data_sources: Optional[list[str]] = None,
        tactile_max_frames: Optional[int] = None,
        include_all_ts: bool = False,
    ):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.data_source_filter = set(data_source_filter) if data_source_filter else None
        self.exclude_data_sources = set(exclude_data_sources) if exclude_data_sources else set()
        self.text_for_label_mode = text_for_label
        self.tactile_label_mode = tactile_label_mode
        self.seed = seed
        self.enable_videos = enable_videos
        self.max_video_frames = max_video_frames
        self.image_patch_size = image_patch_size
        self.video_load_timeout = int(video_load_timeout)
        self.video_suppress_stderr = bool(video_suppress_stderr)
        self.tactile_max_frames = int(tactile_max_frames) if tactile_max_frames else None

        rows: list[dict] = []
        for path in data_files:
            t0 = time.time()
            size_mb = (os.path.getsize(path) / (1024 * 1024)) if os.path.exists(path) else -1
            logger.info("loading %s (%.1f MB)", path, size_mb)
            n_before = len(rows)
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                rows.extend(pq.read_table(path).to_pylist())
            elif path.endswith(".jsonl") or path.endswith(".json"):
                with open(path) as f:
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
            logger.info("  -> %d rows in %.1fs", len(rows) - n_before, time.time() - t0)

        if self.data_source_filter is not None:
            before = len(rows)
            rows = [r for r in rows if r.get("data_source") in self.data_source_filter]
            logger.info("data_source_filter=%s: %d -> %d rows", sorted(self.data_source_filter), before, len(rows))
        if self.exclude_data_sources:
            # Exclude sources before constructing label vocabularies or sampling
            # groups. Excluded tasks must not leak into prototype banks, metrics, or
            # W&B per-class tables.
            rows = [row for row in rows if row.get("data_source") not in self.exclude_data_sources]

        def _has_media(r: dict) -> bool:
            # Content-based routing: a row with a non-empty ``signals`` field is a ts
            # (raw-signal) sample regardless of data_source; otherwise it's img/video.
            if r.get("signals"):
                return True
            if r.get("images"):
                return True
            if self.enable_videos and r.get("videos"):
                return True
            return False

        rows = [r for r in rows if _has_media(r)]
        if not rows:
            raise RuntimeError(
                f"no image/video-bearing samples found in {data_files} (enable_videos={self.enable_videos})"
            )

        # Build label spaces before sampling. Prototype training
        # must compare every anchor against a stable family vocabulary; deriving this
        # from the sampled dataset (or, worse, each minibatch) changes the objective
        # whenever max_samples, world size, or batch composition changes.
        self.ts_label_vocabs: dict[str, tuple[str, ...]] = {}
        vocab_sets: dict[str, set[str]] = {}
        family_rows: dict[str, int] = {}
        for row in rows:
            signals = row.get("signals") or []
            if not signals:
                continue
            family = _signal_family(signals[0])
            label = _text_for_label_encoder(
                row,
                self.text_for_label_mode,
                family=family,
                tactile_label_mode=self.tactile_label_mode,
            )
            family_rows[family] = family_rows.get(family, 0) + 1
            if label:
                vocab_sets.setdefault(family, set()).add(label)
        self.ts_label_vocabs = {family: tuple(sorted(labels)) for family, labels in sorted(vocab_sets.items())}
        for family, labels in self.ts_label_vocabs.items():
            n_rows = family_rows[family]
            logger.info(
                "fixed label vocabulary: family=%s labels=%d rows=%d unique_fraction=%.3f",
                family,
                len(labels),
                n_rows,
                len(labels) / max(1, n_rows),
            )

        if include_all_ts:
            ts_rows = [r for r in rows if r.get("signals")]
            img_rows = [r for r in rows if not r.get("signals")]
            rows = ts_rows + img_rows
            random.Random(self.seed).shuffle(rows)
            logger.info(
                "kept all %d ts + %d img/video rows",
                len(ts_rows),
                len(img_rows),
            )
        elif max_samples and max_samples > 0:
            rows = self._maybe_stratified_sample(rows, max_samples, balanced_sampling_key)

        self.rows = rows
        self.sampling_groups: dict[str, list[int]] = {
            "img": [],
            "smell": [],
            "ecg": [],
            "tactile": [],
        }
        for index, row in enumerate(self.rows):
            signals = row.get("signals") or []
            group = _signal_family(signals[0]) if signals else "img"
            self.sampling_groups[group].append(index)
        logger.info(
            "AlignmentDataset: %d rows from %d files (videos=%s, max_video_frames=%s)",
            len(self.rows),
            len(data_files),
            self.enable_videos,
            self.max_video_frames,
        )

    def _maybe_stratified_sample(self, rows: list[dict], n: int, key: Optional[str]) -> list[dict]:
        """Sample ~``n`` rows balanced across ``key`` buckets, water-filling to hit ``n``.

        Each bucket gets an equal share; buckets smaller than their share are taken whole
        and their leftover quota is redistributed (round by round) to buckets that still
        have rows. This reaches the full target ``n`` even when bucket sizes are very
        uneven (e.g. climb has millions, tactile only ~28k) -- a plain ``n // n_groups``
        split would otherwise undershoot badly.
        """
        rng = random.Random(self.seed)
        if not key:
            rng.shuffle(rows)
            return rows[:n]
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key, "__unknown__")), []).append(r)
        order = sorted(buckets)
        for k in order:
            rng.shuffle(buckets[k])
        total = sum(len(v) for v in buckets.values())
        n = min(n, total)
        pos = {k: 0 for k in order}
        out: list[dict] = []
        remaining = n
        while remaining > 0:
            active = [k for k in order if pos[k] < len(buckets[k])]
            if not active:
                break
            share = max(1, remaining // len(active))
            progressed = False
            for k in active:
                if remaining <= 0:
                    break
                take = min(share, len(buckets[k]) - pos[k], remaining)
                if take <= 0:
                    continue
                out.extend(buckets[k][pos[k] : pos[k] + take])
                pos[k] += take
                remaining -= take
                progressed = True
            if not progressed:
                break
        rng.shuffle(out)
        logger.info("stratified sample: %d from %d (%d groups by %r)", len(out), total, len(order), key)
        return out[:n]

    def __len__(self) -> int:
        return len(self.rows)

    def _none_item(self, idx: int, ds: str) -> dict:
        return {
            "branch": "none",
            "media_kind": "none",
            "media": None,
            "text": "",
            "data_source": ds,
            "index": idx,
        }

    def _load_signal_any(self, sig_entry) -> tuple[Optional[torch.Tensor], str]:
        """Dispatch on the signal ``format`` tag. Returns ``(tensor, ts_format)``.

        ``ts_format`` is ``"smell"``, ``"ecg"``, or ``"tactile"``.
        """
        fmt = sig_entry.get("format") if isinstance(sig_entry, dict) else None
        if fmt == "ts_pt":
            return _load_signal_pt_1d(_signal_path(sig_entry)), _signal_family(sig_entry)
        if fmt == "tactile_pt":
            key = sig_entry.get("key", "glove_right") if isinstance(sig_entry, dict) else "glove_right"
            return _load_tactile_pt(_signal_path(sig_entry), key, self.tactile_max_frames), _signal_family(sig_entry)
        # Default: SmellNet CSV with native channel count and length.
        sig = _load_signal_csv(sig_entry)
        return sig, _signal_family(sig_entry)

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        ds = sample.get("data_source", "")
        # Content-based routing: any row carrying signals -> ts branch.
        branch = "ts" if sample.get("signals") else "img"

        # ts branch: load the RAW signal (no rendered plot). media_kind="signal".
        if branch == "ts":
            signals_meta = sample.get("signals") or []
            if signals_meta:
                sig, ts_format = self._load_signal_any(signals_meta[0])
                if sig is not None:
                    return {
                        "branch": "ts",
                        "media_kind": "signal",
                        "media": sig,
                        "ts_format": ts_format,
                        "text": _text_for_label_encoder(
                            sample,
                            self.text_for_label_mode,
                            family=ts_format,
                            tactile_label_mode=self.tactile_label_mode,
                        ),
                        "data_source": ds,
                        "index": idx,
                    }
            return self._none_item(idx, ds)

        # 1. Prefer images (cheaper, deterministic). img-branch text is unused
        #    (images only feed the distillation loss), so it stays empty.
        images_meta = sample.get("images") or []
        if images_meta:
            pil = _load_image_path_or_dict(images_meta[0])
            if pil is not None:
                return {
                    "branch": branch,
                    "media_kind": "image",
                    "media": pil,
                    "text": "",
                    "data_source": ds,
                    "index": idx,
                }

        # 2. Fall back to videos.
        # Tactile / HB / CLIMB video subsets feed the preservation branch.
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
                    "branch": branch,
                    "media_kind": "video",
                    "media": loaded,
                    "text": "",
                    "data_source": ds,
                    "index": idx,
                }

        return self._none_item(idx, ds)


class FamilyBalancedBatchSampler(Sampler[list[int]]):
    """Yield deterministic per-rank batches with an exact quota per TS family."""

    def __init__(
        self,
        dataset: AlignmentDataset,
        batch_size: int,
        ts_per_family: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ) -> None:
        self.groups = dataset.sampling_groups
        self.batch_size = int(batch_size)
        self.ts_per_family = int(ts_per_family)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0 or self.ts_per_family <= 0:
            raise ValueError("batch_size and ts_per_family must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} must be in [0, {self.world_size})")
        self.img_per_batch = self.batch_size - self.ts_per_family * len(_SIGNAL_FAMILIES)
        if self.img_per_batch < 0:
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than three TS quotas ({self.ts_per_family} each)"
            )
        missing = [family for family in _SIGNAL_FAMILIES if not self.groups.get(family)]
        if missing:
            raise ValueError(f"family-balanced sampling has no rows for {missing}")
        if self.img_per_batch and not self.groups.get("img"):
            raise ValueError(f"batch has {self.img_per_batch} image slots but the dataset has no image rows")

        global_quotas = {
            family: self.ts_per_family * self.world_size for family in _SIGNAL_FAMILIES
        }
        if self.img_per_batch:
            global_quotas["img"] = self.img_per_batch * self.world_size
        self.num_batches = max(
            math.ceil(len(self.groups[family]) / quota) for family, quota in global_quotas.items()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pools = {name: list(indices) for name, indices in self.groups.items() if indices}
        cursors = dict.fromkeys(pools, 0)
        for values in pools.values():
            rng.shuffle(values)

        def draw(name: str, count: int) -> list[int]:
            result: list[int] = []
            while len(result) < count:
                values = pools[name]
                cursor = cursors[name]
                if cursor == len(values):
                    rng.shuffle(values)
                    cursor = 0
                take = min(count - len(result), len(values) - cursor)
                result.extend(values[cursor : cursor + take])
                cursors[name] = cursor + take
            return result

        for batch_index in range(self.num_batches):
            batch: list[int] = []
            for family in _SIGNAL_FAMILIES:
                global_rows = draw(family, self.ts_per_family * self.world_size)
                start = self.rank * self.ts_per_family
                batch.extend(global_rows[start : start + self.ts_per_family])
            if self.img_per_batch:
                global_rows = draw("img", self.img_per_batch * self.world_size)
                start = self.rank * self.img_per_batch
                batch.extend(global_rows[start : start + self.img_per_batch])
            random.Random(self.seed + self.epoch * 1_000_003 + batch_index * 10_007 + self.rank).shuffle(batch)
            yield batch


def collate_alignment(batch: list[dict]) -> dict:
    """Split samples into image/video preservation and time-series buckets."""
    out = {
        "img_image_pil": [],
        "img_video": [],
        "ts_signal": [],
        "ts_format": [],
        "ts_signal_text": [],
        "img_meta": [],
        "ts_meta": [],
    }
    # Single pass: bucket each item by (branch, media_kind). img_meta must list
    # images before videos (the encoder concatenates them in that order), so
    # videos go to a side list and are appended after.
    video_meta: list[dict] = []
    for item in batch:
        kind, media = item["media_kind"], item["media"]
        if media is None:
            continue
        meta = {"data_source": item["data_source"], "index": item["index"], "kind": kind}
        if item["branch"] == "img" and kind == "image":
            out["img_image_pil"].append(media)
            out["img_meta"].append(meta)
        elif item["branch"] == "img" and kind == "video":
            out["img_video"].append(media)
            video_meta.append(meta)
        elif item["branch"] == "ts" and kind == "signal":
            # ts signals keep their native (variable) shape -- the model builds per-sample
            # pseudo-videos and concatenates them (no fixed-shape stacking here).
            out["ts_signal"].append(media)
            out["ts_format"].append(item.get("ts_format", "smell"))
            out["ts_signal_text"].append(item["text"])
            out["ts_meta"].append(meta)
    out["img_meta"].extend(video_meta)
    return out
