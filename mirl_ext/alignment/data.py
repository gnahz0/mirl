# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import logging
import math
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, Dataset, Sampler

logger = logging.getLogger(__name__)

# Set the stable video backend before DataLoader workers import qwen_vl_utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


_SIGNAL_FAMILIES = ("smellnet", "ecg", "tactile")
_EXCLUDED_DATA_SOURCES = {"smellnet_mixture"}


def _label_text(sample: dict) -> str:
    return " ".join(sample["reward_model"]["ground_truth"].split()).casefold()


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
    for row in rows:
        key = None if row.get("signals") else _visual_key(row)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
            row = {
                "images": row.get("images") or [],
                "videos": row.get("videos") or [],
            }
        unique.append(row)
    return unique, len(rows) - len(unique)


def _signal_family(sig_entry: dict) -> str:
    return {"": "smellnet", "ts_pt": "ecg", "tactile_pt": "tactile"}[sig_entry["format"]]


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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tactile = torch.as_tensor(payload["tactile"][key]).float()

    force_raw = payload.get("hand_force_stats")
    if force_raw is None:
        force = tactile.new_empty((tactile.shape[0], 0))
    else:
        force_all = torch.as_tensor(force_raw).float()
        right_idx = [i for i, name in enumerate(payload["hand_force_columns"]) if name.startswith("right_")]
        force = force_all[:, right_idx]
        if force.shape[0] == 1:
            force = force.expand(tactile.shape[0], -1)
        elif force.shape[0] != tactile.shape[0]:
            force = F.interpolate(
                force.t().unsqueeze(0), size=tactile.shape[0], mode="linear", align_corners=False
            ).squeeze(0).t()

    return {"tactile": tactile.contiguous(), "force": force.contiguous()}


class AlignmentDataset(Dataset):
    """Yield lazily loaded image, video, or native-signal samples."""

    def __init__(
        self,
        data_files: list[str],
        max_video_frames: int = 8,
    ):
        self.max_video_frames = max_video_frames

        rows: list[dict] = []
        for path in data_files:
            rows.extend(pq.read_table(path).to_pylist())

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

        self.rows = rows
        self.sampling_groups = {group: [] for group in ("image", "video", *_SIGNAL_FAMILIES)}
        for index, row in enumerate(self.rows):
            signals = row.get("signals") or []
            if signals:
                group = _signal_family(signals[0])
            else:
                group = "image" if row.get("images") else "video"
            self.sampling_groups[group].append(index)
        group_sizes = {name: len(indices) for name, indices in self.sampling_groups.items()}
        logger.info(
            "AlignmentDataset: %d unique rows from %d files; groups=%s",
            len(self.rows),
            len(data_files),
            group_sizes,
        )

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

    def __getitem__(self, idx: int) -> dict:
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
            from verl.utils.dataset.vision_utils import process_image

            return {"kind": "image", "media": process_image(images[0])}

        from verl.utils.dataset.vision_utils import process_video

        video = sample["videos"][0]
        return {
            "kind": "video",
            "media": process_video(
                video,
                image_patch_size=16,
                return_video_metadata=True,
                nframes=self.max_video_frames,
            ),
        }


class HomogeneousBatchSampler(Sampler[list[int]]):
    """Consume every row once in image-, video-, or sensor-only batches."""

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
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        sensor_sizes = [len(self.groups[family]) for family in _SIGNAL_FAMILIES]
        global_batch = batch_size * world_size
        sensor_total = sum(sensor_sizes)
        if sensor_total:
            self.sensor_batches = max(
                math.ceil(sensor_total / global_batch),
                max(math.ceil(size / (ts_per_family * world_size)) for size in sensor_sizes),
            )
            while sum(math.ceil(size / (self.sensor_batches * world_size)) for size in sensor_sizes) > batch_size:
                self.sensor_batches += 1
        else:
            self.sensor_batches = 0
        self.num_batches = self.sensor_batches + sum(
            len(BatchSampler(self.groups[group], global_batch, drop_last=False))
            for group in ("image", "video")
        )

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

        global_batches: list[list[list[int]]] = []
        if self.sensor_batches:
            slot_count = self.sensor_batches * self.world_size
            slots: list[list[int]] = [[] for _ in range(slot_count)]
            for family_index, family in enumerate(_SIGNAL_FAMILIES):
                offset = family_index * slot_count // len(_SIGNAL_FAMILIES)
                for position, row in enumerate(pools[family]):
                    slots[(offset + position) % slot_count].append(row)
            global_batches.extend(BatchSampler(slots, self.world_size, drop_last=False))

        global_batch_size = self.batch_size * self.world_size
        for group in ("image", "video"):
            rows = pools[group]
            batch_count = len(BatchSampler(rows, global_batch_size, drop_last=False))
            slots = [[] for _ in range(batch_count * self.world_size)]
            for position, row in enumerate(rows):
                slots[position % len(slots)].append(row)
            global_batches.extend(BatchSampler(slots, self.world_size, drop_last=False))

        rng.shuffle(global_batches)
        for rank_batches in global_batches:
            batch = rank_batches[self.rank]
            rng.shuffle(batch)
            yield batch


def collate_alignment(batch: list[dict]) -> dict:
    """Bucket media for the preservation and time-series objectives."""
    out = {
        "img_image_pil": [],
        "img_video": [],
        "ts_signal": [],
        "ts_format": [],
        "ts_signal_text": [],
    }
    for item in batch:
        kind = item["kind"]
        if kind == "image":
            out["img_image_pil"].append(item["media"])
        elif kind == "video":
            out["img_video"].append(item["media"])
        else:
            out["ts_signal"].append(item["media"])
            out["ts_format"].append(item["family"])
            out["ts_signal_text"].append(item["text"])
    return out
