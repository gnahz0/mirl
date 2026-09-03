"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import random
from collections import defaultdict

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from mirl_ext.data.schema import (
    TACTILE_NUM_LABELS,
    TACTILE_TASK_LABELS,
    TACTILE_TASK_SPANS,
    media_refs,
    parse_tactile_answer,
    recording_stem,
)
from mirl_ext.data.signals import load_signal
from mirl_ext.data.signals import signal_family as _signal_family

# Metric reduction order must be identical on every rank, so this tuple is the one
# definition of it; tactile aligns against TASK_LABELS and never has a vocabulary.
_TS_FAMILIES: tuple[str, ...] = ("ecg", "tactile")
_CLASSIFICATION_FAMILIES: tuple[str, ...] = ("ecg",)

# Compatibility aliases for alignment call sites. Their values are derived once
# from ``TactileTaskSpec`` in data.schema rather than reconstructed here.
TASK_LABELS = TACTILE_TASK_LABELS
TACTILE_SPANS = TACTILE_TASK_SPANS


def _recording_stem(row: dict) -> str:
    """Resolve the shared haptic/QA recording identifier."""
    stem = recording_stem(row)
    if stem is None:
        raise ValueError("tactile row has neither extra_info.stem nor extra_info.video_path")
    return stem


def _parse_annotation_answer(answer: object, task: str) -> tuple[int, ...]:
    """Compatibility wrapper for the shared schema parser."""
    return parse_tactile_answer(answer, task)


def _media_kind(row: dict) -> str:
    """Return the row's sole media kind; reject missing or ambiguous rows."""
    present = tuple(
        kind for kind, field in (("signal", "signals"), ("image", "images"), ("video", "videos")) if row.get(field)
    )
    if len(present) != 1:
        raise ValueError(
            "alignment row must contain exactly one of signals, images, or videos; "
            f"found {present or 'none'} for data_source={row.get('data_source')!r}"
        )
    return present[0]


def _signal_entry(row: dict) -> dict:
    signals = row.get("signals") or []
    if len(signals) != 1:
        raise ValueError(f"alignment signal row needs exactly one signals[] entry, got {len(signals)}")
    return signals[0]


def _annotation_targets(
    rows: list[dict],
    valid_stems: set[str],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Join the fixed closed QA answers by recording stem."""
    answers: dict[str, dict[str, set[tuple[int, ...]]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        task = str(row["data_source"])
        if task not in TASK_LABELS:
            continue
        stem = _recording_stem(row)
        if stem not in valid_stems:
            continue
        answers[stem][task].add(_parse_annotation_answer(row["reward_model"]["ground_truth"], task))
    # Disagreeing duplicate QA rows drop the task, not the recording: other tasks on the stem still supervise.
    return {
        stem: {task: next(iter(choices)) for task, choices in tasks.items() if len(choices) == 1}
        for stem, tasks in answers.items()
    }


class AlignmentDataset(Dataset):
    """Yield lazily loaded image, video, or native-signal samples."""

    def __init__(
        self,
        data_files: list[str],
        sample_media_kinds: tuple[str, ...] = ("signal", "image", "video"),
    ):
        rows: list[dict] = []
        for path in data_files:
            rows.extend(pq.read_table(path).to_pylist())

        tactile_rows = [
            row for row in rows if _media_kind(row) == "signal" and _signal_family(_signal_entry(row)) == "tactile"
        ]
        if tactile_rows:
            stems = [_recording_stem(row) for row in tactile_rows]
            annotations = _annotation_targets(rows, set(stems))
            unannotated = sorted({stem for stem in stems if not annotations.get(stem)})
            if unannotated:
                preview = ", ".join(unannotated[:5])
                raise ValueError(
                    f"{len(unannotated)} tactile signal recording(s) have no unambiguous "
                    f"closed-task annotations; first: {preview}"
                )
            for row, stem in zip(tactile_rows, stems, strict=True):
                row["_tactile_targets"] = annotations[stem]

        allowed = frozenset(sample_media_kinds)
        unknown = allowed.difference(("signal", "image", "video"))
        if unknown:
            raise ValueError(f"unsupported sampled media kinds: {tuple(sorted(unknown))}")
        rows = [row for row in rows if _media_kind(row) in allowed]

        self.rows = rows
        # Tactile rows are supervised by their joined QA spans, never by a vocabulary.
        vocab_sets: dict[str, set[str]] = {family: set() for family in _CLASSIFICATION_FAMILIES}
        self.sampling_groups: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(rows):
            kind = _media_kind(row)
            if kind == "signal":
                if (family := _signal_family(_signal_entry(row))) in vocab_sets:
                    vocab_sets[family].add(row["reward_model"]["ground_truth"])
            self.sampling_groups.setdefault((kind, row["data_source"]), []).append(index)
        self.ts_label_vocabs = {family: tuple(sorted(labels)) for family, labels in vocab_sets.items() if labels}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        kind = _media_kind(sample)
        if kind == "signal":
            media, family = load_signal(_signal_entry(sample))
            return {
                "kind": "signal",
                "media": media,
                "family": family,
                "text": sample["reward_model"]["ground_truth"],
                **({"targets": sample["_tactile_targets"]} if "_tactile_targets" in sample else {}),
            }
        images, video_path = media_refs(sample)
        if kind == "image":
            if not images:
                raise ValueError("image row has no usable image path")
            return {"kind": "image", "media": images[0]}
        if kind == "video":
            if not video_path:
                raise ValueError("video row has no usable video path")
            return {
                "kind": "video",
                "media": video_path,
                "family": str(sample["data_source"]),
            }
        raise ValueError(f"unsupported alignment media kind {kind!r}")


class HomogeneousBatchSampler(Sampler[list[int]]):
    """Yield source-homogeneous batches with one nonempty shard per rank; repeated
    signal sources take complete shuffled passes, everything else stays one-pass."""

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

        # Count through the same chunking as __iter__ so __len__ never drifts.
        effective_sizes = {
            group: len(rows) * self.signal_repeat_factors.get(group[1], 1) for group, rows in self.groups.items()
        }
        self.num_batches = len(self._global_batches({g: list(range(n)) for g, n in effective_sizes.items()}))

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
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pools: dict[tuple[str, str], list[int]] = {}
        for group, indices in self.groups.items():
            values: list[int] = []
            for _ in range(self.signal_repeat_factors.get(group[1], 1)):
                cycle = list(indices)
                rng.shuffle(cycle)
                values.extend(cycle)
            pools[group] = values

        global_batches = self._global_batches(pools)
        rng.shuffle(global_batches)
        for global_batch in global_batches:
            yield global_batch[self.rank :: self.world_size]


def collate_alignment(batch: list[dict]) -> dict:
    """Keep variable-shape media as one source-homogeneous list."""
    if not batch:
        raise ValueError("cannot collate an empty alignment batch")
    kind = batch[0]["kind"]
    if kind not in {"signal", "image", "video"}:
        raise ValueError(f"unsupported alignment media kind {kind!r}")
    if any(item["kind"] != kind for item in batch):
        raise ValueError("alignment batch mixes media kinds")
    if kind in {"signal", "video"} and any(item.get("family") != batch[0].get("family") for item in batch):
        raise ValueError(f"alignment batch mixes {kind} families")
    collated = {
        "kind": kind,
        "media": [item["media"] for item in batch],
        "family": batch[0].get("family"),
        "text": [item["text"] for item in batch] if kind == "signal" else [],
    }
    if kind == "signal" and "targets" in batch[0]:
        targets = torch.zeros(len(batch), TACTILE_NUM_LABELS, dtype=torch.float32)
        masks = torch.zeros(len(batch), TACTILE_NUM_LABELS, dtype=torch.float32)
        for row, item in enumerate(batch):
            for task, (start, stop) in TACTILE_SPANS.items():
                masks[row, start:stop] = float(task in item["targets"])
                targets[row, [start + index for index in item["targets"].get(task, ())]] = 1.0
        collated["targets"] = targets
        collated["masks"] = masks
    return collated
