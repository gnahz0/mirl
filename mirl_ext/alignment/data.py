# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler

logger = logging.getLogger(__name__)

# Select the video backend before worker imports.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


_SIGNAL_FAMILIES = ("smellnet", "ecg", "tactile")


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
                "data_source": row["data_source"],
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

        rows = [row for row in rows if row["data_source"] != "smellnet_mixture"]
        rows, removed = _deduplicate_visual_rows(rows)
        if removed:
            logger.info("removed %d repeated image/video annotation rows", removed)

        vocab_sets = {family: set() for family in _SIGNAL_FAMILIES}
        for row in rows:
            signals = row.get("signals") or []
            if not signals:
                continue
            family = _signal_family(signals[0])
            vocab_sets[family].add(row["reward_model"]["ground_truth"])
        self.ts_label_vocabs = {
            family: tuple(sorted(vocab_sets[family]))
            for family in _SIGNAL_FAMILIES
            if vocab_sets[family]
        }

        self.rows = rows
        self.sampling_groups: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(self.rows):
            signals = row.get("signals") or []
            if signals:
                kind = "signal"
            else:
                kind = "image" if row.get("images") else "video"
            group = (kind, row["data_source"])
            self.sampling_groups.setdefault(group, []).append(index)
        group_sizes = {
            f"{kind}/{source}": len(indices)
            for (kind, source), indices in self.sampling_groups.items()
        }
        logger.info(
            "AlignmentDataset: %d unique rows from %d files; groups=%s",
            len(self.rows),
            len(data_files),
            group_sizes,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_signal(self, sig_entry: dict) -> tuple[torch.Tensor, str]:
        family = _signal_family(sig_entry)
        path = sig_entry["signal"]
        if family == "ecg":
            signal = torch.load(path, map_location="cpu", weights_only=False)
            return signal.float().contiguous(), family
        if family == "tactile":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            signal = torch.as_tensor(payload["tactile"][sig_entry["key"]])
            return signal.float().contiguous(), family
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
                "text": sample["reward_model"]["ground_truth"],
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
    """Yield source-homogeneous batches with one nonempty shard per rank."""

    def __init__(
        self,
        dataset: AlignmentDataset,
        batch_size: int,
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
        self.num_batches = len(self._global_batches(self.groups))
        dropped = sum(
            len(BatchSampler(rows, batch_size * world_size, drop_last=False))
            for rows in self.groups.values()
        ) - self.num_batches
        if dropped:
            logger.warning(
                "HomogeneousBatchSampler: dropped %d global batch(es) with fewer than "
                "world_size=%d rows; a group that small cannot give every rank a row.",
                dropped,
                world_size,
            )

    def _global_batches(self, pools: dict) -> list[list[int]]:
        """Build global batches that give every rank a sample."""
        batches: list[list[int]] = []
        global_batch_size = self.batch_size * self.world_size
        for rows in pools.values():
            batch_count = len(BatchSampler(rows, global_batch_size, drop_last=False))
            for start in range(batch_count):
                chunk = rows[start * len(rows) // batch_count : (start + 1) * len(rows) // batch_count]
                if len(chunk) >= self.world_size:
                    batches.append(chunk)
        return batches

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

        global_batches = self._global_batches(pools)
        rng.shuffle(global_batches)
        for global_batch in global_batches:
            yield global_batch[self.rank :: self.world_size]


def collate_alignment(batch: list[dict]) -> dict:
    """Keep variable-shape media as one source-homogeneous list."""
    kind = batch[0]["kind"]
    return {
        "kind": kind,
        "media": [item["media"] for item in batch],
        "family": batch[0].get("family"),
        "text": [item["text"] for item in batch] if kind == "signal" else [],
    }
