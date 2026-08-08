# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Native tactile loading and source-homogeneous batching."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

TASK_LABELS: dict[str, tuple[str, ...]] = {
    "initial_fingers": (
        "initial contact: thumb",
        "initial contact: index finger",
        "initial contact: middle finger",
        "initial contact: ring finger",
        "initial contact: pinky finger",
        "initial contact: palm",
    ),
    "highest_pressure": (
        "highest pressure: thumb",
        "highest pressure: index finger",
        "highest pressure: middle finger",
        "highest pressure: ring finger",
        "highest pressure: pinky finger",
        "highest pressure: palm",
    ),
    "force_level": (
        "force level: light, under 5 newtons",
        "force level: moderate, 5 to 10 newtons",
        "force level: firm, 10 to 20 newtons",
        "force level: strong, over 20 newtons",
    ),
    "grip_stability": (
        "grip stability: stable",
        "grip stability: unstable",
    ),
    "contact_feature": (
        "contact geometry: edge",
        "contact geometry: flat surface",
        "contact geometry: curved surface",
        "contact geometry: corner",
        "contact geometry: multiple edges",
        "contact geometry: edge and surface",
        "contact geometry: transitioning from edge to surface",
        "contact geometry: complex geometry with multiple features",
    ),
    "local_shape": (
        "local surface shape: flat",
        "local surface shape: convex",
        "local surface shape: concave",
        "local surface shape: edge",
    ),
}
MULTILABEL_TASKS = frozenset(("initial_fingers", "highest_pressure"))
logger = logging.getLogger(__name__)


def _rewrite_path(path: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    for old, new in rewrites:
        if path == old or path.startswith(f"{old}/"):
            return f"{new}{path[len(old) :]}"
    return path


def _rewrite_media_paths(row: dict, rewrites: tuple[tuple[str, str], ...]) -> int:
    changed = 0
    for entry in row.get("signals") or []:
        for key in ("signal", "path"):
            path = entry.get(key)
            if path:
                rewritten = _rewrite_path(str(path), rewrites)
                entry[key] = rewritten
                changed += rewritten != path
    return changed


def _extra_info(row: dict) -> dict:
    value = row.get("extra_info") or {}
    return json.loads(value) if isinstance(value, str) else value


def _recording_stem(row: dict) -> str:
    extra = _extra_info(row)
    return str(extra.get("stem") or Path(str(extra["video_path"])).stem)


def _annotation_targets(
    annotation_files: list[str],
    tasks: tuple[str, ...],
    valid_stems: set[str],
) -> tuple[dict[str, dict[str, tuple[int, ...]]], dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, list[tuple[int, ...]]]] = defaultdict(lambda: defaultdict(list))
    for path in annotation_files:
        table = pq.read_table(path, columns=["data_source", "reward_model", "extra_info"])
        for row in table.to_pylist():
            task = str(row["data_source"])
            if task not in tasks:
                continue
            stem = _recording_stem(row)
            if stem not in valid_stems:
                continue
            answer = str(row["reward_model"]["ground_truth"])
            indices = tuple(sorted({ord(letter.strip()) - ord("A") for letter in answer.split(",")}))
            grouped[stem][task].append(indices)

    resolved: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    audit = {task: {"duplicates": 0, "conflicts": 0} for task in tasks}
    for stem, task_rows in grouped.items():
        for task, answers in task_rows.items():
            if len(answers) > 1:
                audit[task]["duplicates"] += 1
            unique = set(answers)
            if len(unique) == 1:
                resolved[stem][task] = unique.pop()
            else:
                audit[task]["conflicts"] += 1
    return dict(resolved), audit


class AlignmentDataset(Dataset):
    def __init__(
        self,
        data_files: list[str],
        annotation_files: list[str],
        tasks: list[str],
        path_rewrites: dict[str, str] | None = None,
    ):
        self.tasks = tuple(tasks)
        unknown = set(self.tasks) - TASK_LABELS.keys()
        if unknown:
            raise ValueError(f"unknown tactile tasks: {sorted(unknown)}")

        rows = [row for path in data_files for row in pq.read_table(path).to_pylist()]
        valid_stems = {_recording_stem(row) for row in rows}
        rewrites = tuple(
            sorted(
                ((str(old).rstrip("/"), str(new).rstrip("/")) for old, new in (path_rewrites or {}).items()),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
        )
        if rewrites:
            sum(_rewrite_media_paths(row, rewrites) for row in rows)

        annotations, audit = _annotation_targets(annotation_files, self.tasks, valid_stems)
        self.rows = []
        for row in rows:
            targets = annotations.get(_recording_stem(row), {})
            if targets:
                row["_targets"] = targets
                self.rows.append(row)

        self.task_labels = {task: TASK_LABELS[task] for task in self.tasks}
        self.task_positive_rates: dict[str, float] = {}
        for task, labels in self.task_labels.items():
            observed = [row["_targets"][task] for row in self.rows if task in row["_targets"]]
            positives = sum(len(target) for target in observed)
            self.task_positive_rates[task] = positives / (len(observed) * len(labels))
            logger.info(
                "tactile labels: task=%s observed=%d missing=%d duplicates=%d conflicts=%d positive_rate=%.4f",
                task,
                len(observed),
                len(self.rows) - len(observed),
                audit[task]["duplicates"],
                audit[task]["conflicts"],
                self.task_positive_rates[task],
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        signal = row["signals"][0]
        payload = torch.load(signal["signal"], map_location="cpu", weights_only=False)
        tactile = torch.as_tensor(payload["tactile"][signal["key"]]).float().contiguous()
        return {
            "media": tactile,
            "targets": row["_targets"],
            "tasks": self.tasks,
        }


class HomogeneousBatchSampler(Sampler[list[int]]):
    """Shuffle one tactile pass and give each rank a nonempty batch shard."""

    def __init__(
        self,
        dataset: AlignmentDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ) -> None:
        self.num_samples = len(dataset)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        global_batch_size = self.batch_size * self.world_size
        self.num_batches = (self.num_samples + global_batch_size - 1) // global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        indices = list(range(self.num_samples))
        rng.shuffle(indices)
        for batch_index in range(self.num_batches):
            start = batch_index * self.num_samples // self.num_batches
            end = (batch_index + 1) * self.num_samples // self.num_batches
            global_batch = indices[start:end]
            yield global_batch[self.rank :: self.world_size]


def collate_alignment(batch: list[dict]) -> dict:
    task_targets: dict[str, torch.Tensor] = {}
    task_masks: dict[str, torch.Tensor] = {}
    for task in batch[0]["tasks"]:
        labels = TASK_LABELS[task]
        targets = torch.zeros((len(batch), len(labels)), dtype=torch.float32)
        mask = torch.zeros(len(batch), dtype=torch.bool)
        for row_index, item in enumerate(batch):
            positive = item["targets"].get(task)
            if positive is not None:
                targets[row_index, list(positive)] = 1.0
                mask[row_index] = True
        task_targets[task] = targets
        task_masks[task] = mask

    return {
        "media": [item["media"] for item in batch],
        "targets": task_targets,
        "masks": task_masks,
    }
