# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import contextlib
import heapq
import logging
import os
import random
import signal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# Set the stable video backend before DataLoader workers import qwen_vl_utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


_SIGNAL_FAMILIES = ("smell", "ecg", "tactile")
_EXCLUDED_DATA_SOURCES = {"smellnet_mixture"}


def _canonical_label_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _label_text(sample: dict) -> str:
    return _canonical_label_text(sample["reward_model"]["ground_truth"])


def _visual_key(row: dict) -> tuple[str, str] | None:
    """Identify the physical image or video while ignoring its QA annotation."""
    for column, path_key in (("images", "image"), ("videos", "video")):
        entries = row.get(column) or []
        if not entries:
            continue
        entry = entries[0]
        if isinstance(entry, str):
            return column, entry
        path = entry.get(path_key) or entry.get("path")
        return (column, str(path)) if path else None
    return None


def _deduplicate_visual_rows(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    removed = 0
    for row in rows:
        key = None if row.get("signals") else _visual_key(row)
        if key is not None and key in seen:
            removed += 1
            continue
        if key is not None:
            seen.add(key)
            row = {
                "data_source": row["data_source"],
                "images": row.get("images") or [],
                "videos": row.get("videos") or [],
            }
        unique.append(row)
    return unique, removed


def _load_image(img_entry) -> Image.Image | None:
    try:
        from verl.utils.dataset.vision_utils import process_image

        return process_image(img_entry)
    except Exception:  # noqa: BLE001
        return None


def _signal_family(sig_entry: dict) -> str:
    return {"": "smell", "ts_pt": "ecg", "tactile_pt": "tactile"}[sig_entry["format"]]


def _load_signal_csv(path: str) -> torch.Tensor:
    with open(path) as f:
        header = f.readline().strip().split(",")
    keep_idx = [i for i, name in enumerate(header) if "time" not in name.casefold()]
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=keep_idx,
        dtype=np.float32,
    )
    return torch.from_numpy(np.ascontiguousarray(data.T))


def _load_tactile_pt(
    path: str,
    key: str,
) -> dict[str, torch.Tensor]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = torch.as_tensor(d["tactile"][key]).float()

    force_raw = d.get("hand_force_stats")
    if force_raw is None:
        force = t.new_empty((t.shape[0], 0))
    else:
        force_all = torch.as_tensor(force_raw).float()
        right_idx = [i for i, name in enumerate(d["hand_force_columns"]) if name.startswith("right_")]
        force = force_all[:, right_idx]
        if force.shape[0] == 1:
            force = force.expand(t.shape[0], -1)
        elif force.shape[0] != t.shape[0]:
            force = F.interpolate(
                force.t().unsqueeze(0), size=t.shape[0], mode="linear", align_corners=False
            ).squeeze(0).t()

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


def _load_video_entry(video_entry: dict, max_frames: int):
    from verl.utils.dataset.vision_utils import process_video

    old_handler = signal.signal(signal.SIGALRM, _video_timeout_handler)
    signal.alarm(30)
    try:
        with _suppress_fd_stderr():
            return process_video(
                video_entry,
                image_patch_size=16,
                return_video_metadata=True,
                nframes=max_frames,
            )
    except _VideoTimeout:
        logger.warning("video decode timed out after 30s: %s", video_entry)
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
        data_files: list[str],
        max_samples: int = -1,
        seed: int = 42,
        max_video_frames: int = 8,
        data_source_filter: list[str] | None = None,
    ):
        self.seed = seed
        self.max_video_frames = max_video_frames

        rows: list[dict] = []
        for path in data_files:
            rows.extend(pq.read_table(path).to_pylist())

        if data_source_filter:
            sources = set(data_source_filter)
            rows = [row for row in rows if row["data_source"] in sources]

        rows = [row for row in rows if row["data_source"] not in _EXCLUDED_DATA_SOURCES]
        rows, removed = _deduplicate_visual_rows(rows)
        if removed:
            logger.info("removed %d repeated image/video annotation rows", removed)

        vocab_sets = {family: set() for family in _SIGNAL_FAMILIES}
        for row in rows:
            signals = row.get("signals") or []
            if not signals:
                continue
            family = _signal_family(signals[0])
            vocab_sets[family].add(_label_text(row))
        self.ts_label_vocabs = {
            family: tuple(sorted(vocab_sets[family]))
            for family in _SIGNAL_FAMILIES
            if vocab_sets[family]
        }

        if max_samples > 0:
            rows = self._balanced_sample(rows, max_samples)
        else:
            random.Random(seed).shuffle(rows)

        self.rows = rows
        self.sampling_groups = {family: [] for family in ("img", *_SIGNAL_FAMILIES)}
        for index, row in enumerate(self.rows):
            signals = row.get("signals") or []
            group = _signal_family(signals[0]) if signals else "img"
            self.sampling_groups[group].append(index)
        group_sizes = {name: len(indices) for name, indices in self.sampling_groups.items()}
        logger.info(
            "AlignmentDataset: %d unique rows from %d files; groups=%s",
            len(self.rows),
            len(data_files),
            group_sizes,
        )

    def _balanced_sample(self, rows: list[dict], n: int) -> list[dict]:
        rng = random.Random(self.seed)
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row["data_source"], []).append(row)
        order = sorted(buckets)
        for name in order:
            rng.shuffle(buckets[name])

        out: list[dict] = []
        target = min(n, len(rows))
        while len(out) < target:
            for name in order:
                if buckets[name]:
                    out.append(buckets[name].pop())
                if len(out) == target:
                    break
        rng.shuffle(out)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def _load_signal(self, sig_entry: dict) -> tuple[torch.Tensor | dict[str, torch.Tensor], str]:
        family = _signal_family(sig_entry)
        path = sig_entry["signal"]
        if family == "ecg":
            signal = torch.load(path, map_location="cpu", weights_only=False)
            return signal.float().contiguous(), family
        if family == "tactile":
            return _load_tactile_pt(path, sig_entry["key"]), family
        return _load_signal_csv(path), family

    def __getitem__(self, idx: int) -> dict | None:
        sample = self.rows[idx]
        signals = sample.get("signals") or []
        if signals:
            media, family = self._load_signal(signals[0])
            return {
                "kind": "signal",
                "media": media,
                "family": family,
                "text": _label_text(sample),
            }

        images = sample.get("images") or []
        if images:
            pil = _load_image(images[0])
            if pil is not None:
                return {"kind": "image", "media": pil}

        videos = sample.get("videos") or []
        if videos:
            loaded = _load_video_entry(videos[0], self.max_video_frames)
            if loaded is not None:
                return {"kind": "video", "media": loaded}

        return None


class FamilyBalancedBatchSampler(Sampler[list[int]]):
    """Consume every row once and spread small sensor families across the epoch."""

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
        self.ts_per_family = ts_per_family
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        if ts_per_family * len(_SIGNAL_FAMILIES) > batch_size:
            raise ValueError("sensor family quotas exceed the per-rank batch size")

        global_family_quota = ts_per_family * world_size
        family_batches = max(
            (len(self.groups[family]) + global_family_quota - 1) // global_family_quota
            for family in _SIGNAL_FAMILIES
        )
        total_rows = sum(len(indices) for indices in self.groups.values())
        global_batch = batch_size * world_size
        full_batches = (total_rows + global_batch - 1) // global_batch
        self.num_batches = max(family_batches, full_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        epoch_seed = self.seed + self.epoch * 1_000_003
        rng = random.Random(epoch_seed)
        pools = {name: list(indices) for name, indices in self.groups.items()}
        for values in pools.values():
            rng.shuffle(values)

        slot_count = self.num_batches * self.world_size
        batches: list[list[int]] = [[] for _ in range(slot_count)]
        for family in _SIGNAL_FAMILIES:
            rows = pools[family]
            for position, row in enumerate(rows):
                slot = position * slot_count // len(rows)
                batches[slot].append(row)

        slot_order = list(range(slot_count))
        rng.shuffle(slot_order)
        tie_break = {slot: order for order, slot in enumerate(slot_order)}
        available = [
            (len(batch), tie_break[slot], slot)
            for slot, batch in enumerate(batches)
            if len(batch) < self.batch_size
        ]
        heapq.heapify(available)
        for row in pools["img"]:
            size, tie, slot = heapq.heappop(available)
            batches[slot].append(row)
            if size + 1 < self.batch_size:
                heapq.heappush(available, (size + 1, tie, slot))

        for batch_index in range(self.num_batches):
            batch = batches[batch_index * self.world_size + self.rank]
            rng.shuffle(batch)
            yield batch


def collate_alignment(batch: list[dict | None]) -> dict:
    """Bucket media for the preservation and time-series objectives."""
    out = {
        "img_image_pil": [],
        "img_video": [],
        "ts_signal": [],
        "ts_format": [],
        "ts_signal_text": [],
    }
    for item in batch:
        if item is None:
            continue
        kind = item["kind"]
        if kind == "image":
            out["img_image_pil"].append(item["media"])
        elif kind == "video":
            out["img_video"].append(item["media"])
        elif kind == "signal":
            out["ts_signal"].append(item["media"])
            out["ts_format"].append(item["family"])
            out["ts_signal_text"].append(item["text"])
    return out
