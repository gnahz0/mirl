# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Native tactile loading and source-homogeneous batching."""

from __future__ import annotations

import logging
import random

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

_SIGNAL_FAMILIES = ("tactile",)
logger = logging.getLogger(__name__)


def _rewrite_path(path: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    """Replace the longest matching path prefix."""
    for old, new in rewrites:
        if path == old or path.startswith(f"{old}/"):
            return f"{new}{path[len(old):]}"
    return path


def _rewrite_media_paths(row: dict, rewrites: tuple[tuple[str, str], ...]) -> int:
    """Translate embedded tactile paths in one Parquet row."""
    changed = 0
    for entry in row.get("signals") or []:
        for key in ("signal", "path"):
            path = entry.get(key)
            if not path:
                continue
            rewritten = _rewrite_path(str(path), rewrites)
            entry[key] = rewritten
            changed += rewritten != path
    return changed


class AlignmentDataset(Dataset):
    def __init__(
        self,
        data_files: list[str],
        path_rewrites: dict[str, str] | None = None,
    ):
        self.rows = [
            row
            for path in data_files
            for row in pq.read_table(path).to_pylist()
        ]
        rewrites = tuple(
            sorted(
                ((str(old).rstrip("/"), str(new).rstrip("/")) for old, new in (path_rewrites or {}).items()),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
        )
        if rewrites:
            sum(_rewrite_media_paths(row, rewrites) for row in self.rows)
        labels = {row["reward_model"]["ground_truth"] for row in self.rows}
        self.ts_label_vocabs = {"tactile": tuple(sorted(labels))}
        self.sampling_groups = {
            ("signal", "haptic_tactile"): list(range(len(self.rows)))
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        signal = row["signals"][0]
        payload = torch.load(signal["signal"], map_location="cpu", weights_only=False)
        tactile = torch.as_tensor(payload["tactile"][signal["key"]]).float().contiguous()
        return {
            "kind": "signal",
            "media": tactile,
            "family": "tactile",
            "text": row["reward_model"]["ground_truth"],
        }


class HomogeneousBatchSampler(Sampler[list[int]]):
    """Yield source-homogeneous batches with one nonempty shard per rank.

    Training may repeat complete shuffled passes over selected signal sources.
    Visual sources and omitted signal sources always retain one-pass sampling.
    """

    def __init__(
        self,
        dataset: AlignmentDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        signal_repeat_factors: dict[str, int] | None = None,
    ) -> None:
        self.groups = dataset.sampling_groups
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        self.signal_repeat_factors = dict(signal_repeat_factors or {})

        signal_sources = {source for kind, source in self.groups if kind == "signal"}
        unknown = sorted(set(self.signal_repeat_factors) - signal_sources)
        if unknown:
            raise ValueError(f"signal_repeat_factors contains unknown signal sources: {unknown}")
        for source, factor in self.signal_repeat_factors.items():
            if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
                raise ValueError(
                    "signal_repeat_factors values must be positive integers; "
                    f"got {source}={factor!r}"
                )

        effective_sizes = {
            group: len(rows) * self._repeat_factor(group)
            for group, rows in self.groups.items()
        }
        global_batch_size = self.batch_size * self.world_size
        batch_counts = {
            group: (size + global_batch_size - 1) // global_batch_size
            for group, size in effective_sizes.items()
        }
        self.num_batches = sum(
            count
            for group, count in batch_counts.items()
            if effective_sizes[group] >= self.world_size
        )
        dropped = sum(batch_counts.values()) - self.num_batches
        if dropped:
            logger.warning(
                "HomogeneousBatchSampler: dropped %d global batch(es) with fewer than "
                "world_size=%d rows; a group that small cannot give every rank a row.",
                dropped,
                world_size,
            )
        if self.signal_repeat_factors:
            logger.info(
                "HomogeneousBatchSampler: signal repeat factors=%s; effective rows=%s",
                self.signal_repeat_factors,
                {
                    f"{kind}/{source}": effective_sizes[(kind, source)]
                    for kind, source in self.groups
                    if kind == "signal"
                },
            )

    def _repeat_factor(self, group: tuple[str, str]) -> int:
        kind, source = group
        return self.signal_repeat_factors.get(source, 1) if kind == "signal" else 1

    def _global_batches(self, pools: dict) -> list[list[int]]:
        """Build global batches that give every rank a sample."""
        batches: list[list[int]] = []
        global_batch_size = self.batch_size * self.world_size
        for rows in pools.values():
            batch_count = (len(rows) + global_batch_size - 1) // global_batch_size
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
        pools: dict[tuple[str, str], list[int]] = {}
        for group, indices in self.groups.items():
            values: list[int] = []
            for _ in range(self._repeat_factor(group)):
                cycle = list(indices)
                rng.shuffle(cycle)
                values.extend(cycle)
            pools[group] = values

        global_batches = self._global_batches(pools)
        rng.shuffle(global_batches)
        for global_batch in global_batches:
            yield global_batch[self.rank :: self.world_size]


def collate_alignment(batch: list[dict]) -> dict:
    return {
        "kind": "signal",
        "media": [item["media"] for item in batch],
        "family": "tactile",
        "text": [item["text"] for item in batch],
        "skipped": {},
    }
