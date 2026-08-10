# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Lazy media and native-signal loading for Stage-1 alignment."""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

# Select the video backend before worker imports.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


# Metric reduction order must be identical on every rank, so this tuple is the one
# definition of it; tactile aligns against TASK_LABELS and never has a vocabulary.
_TS_FAMILIES: tuple[str, ...] = ("smellnet", "ecg", "tactile")
_CLASSIFICATION_FAMILIES: tuple[str, ...] = ("smellnet", "ecg")
_SIGNAL_FORMATS = {"": "smellnet", "ts_pt": "ecg", "tactile_pt": "tactile"}
_SKIPPABLE_LOAD_ERRORS = (OSError, ValueError, RuntimeError, EOFError)

# Closed-label tactile tasks. The haptic signal Parquets contain open-ended
# descriptions; these labels are joined from the tactile QA Parquets by the
# recording stem. SFT/RL should keep the original ordered QA rows instead.
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

# Column span of each task inside the concatenated tactile bank. The six tasks are
# independent Bernoulli decisions under the sigmoid objective -- nothing couples
# them -- so one bank of TACTILE_NUM_LABELS entries is equivalent to six banks, and
# anything wanting a single task slices these columns instead of holding its own.
TACTILE_SPANS: dict[str, tuple[int, int]] = {}
_offset = 0
for _task, _task_labels in TASK_LABELS.items():
    TACTILE_SPANS[_task] = (_offset, _offset + len(_task_labels))
    _offset += len(_task_labels)
TACTILE_NUM_LABELS = _offset
del _offset, _task, _task_labels


def _recording_stem(row: dict) -> str:
    """Resolve the shared haptic/QA recording identifier."""
    extra = row.get("extra_info") or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    stem = extra.get("stem")
    if stem:
        return str(stem)
    video_path = extra.get("video_path")
    if not video_path:
        raise ValueError("tactile row has neither extra_info.stem nor extra_info.video_path")
    return Path(str(video_path)).stem


def _parse_annotation_answer(answer: object, task: str) -> tuple[int, ...]:
    """Convert comma-separated QA choices (for example ``A,B,F``) to IDs."""
    labels = TASK_LABELS[task]
    choices = [choice.strip() for choice in str(answer).split(",")]
    if any(len(choice) != 1 or not choice.isalpha() for choice in choices):
        raise ValueError(f"invalid answer {answer!r}")
    # Splitting always yields a choice and no alphabetic codepoint sits below "A", so
    # the upper bound is the only reachable range failure -- it also rejects lowercase.
    indices = tuple(sorted({ord(choice) - ord("A") for choice in choices}))
    if indices[-1] >= len(labels):
        raise ValueError(f"answer {answer!r} is outside A-{chr(ord('A') + len(labels) - 1)}")
    if task not in MULTILABEL_TASKS and len(indices) != 1:
        raise ValueError(f"exclusive task has {len(indices)} choices in {answer!r}")
    return indices


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
    # Duplicate QA rows that disagree drop the pair, not the recording: the stem stays
    # a key so its other tasks still supervise.
    return {
        stem: {task: next(iter(choices)) for task, choices in tasks.items() if len(choices) == 1}
        for stem, tasks in answers.items()
    }


def _visual_entries(row: dict) -> list[tuple[str, tuple[str, str], dict | str]]:
    """Return every physical image/video entry and its path-based key."""
    media: list[tuple[str, tuple[str, str], dict | str]] = []
    for column, path_key in (("images", "image"), ("videos", "video")):
        entries = row.get(column) or []
        for entry in entries:
            if isinstance(entry, str):
                path = entry
            elif isinstance(entry, dict):
                path = entry.get(path_key) or entry.get("path")
            else:
                continue
            if path:
                media.append((column, (column, str(path)), entry))
    return media


def _expand_and_deduplicate_visual_rows(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Make one Stage-1 anchor per unique physical image or video path."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    repeated = 0
    for row in rows:
        if row.get("signals"):
            unique.append(row)
            continue

        # Stage 1 ignores QA text, so each physical item is an independent
        # preservation anchor. SFT/RL must instead retain the complete ordered
        # media list so it remains aligned with the prompt placeholders.
        entries = _visual_entries(row)
        if not entries:
            unique.append(row)
            continue
        for column, key, entry in entries:
            if key in seen:
                repeated += 1
                continue
            seen.add(key)
            unique.append(
                {
                    "data_source": row["data_source"],
                    "images": [entry] if column == "images" else [],
                    "videos": [entry] if column == "videos" else [],
                }
            )
    return unique, len(seen), repeated


def _signal_family(sig_entry: dict) -> str:
    return _SIGNAL_FORMATS[sig_entry["format"]]


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


def _factor_aligned_video_size(height: int, width: int, factor: int = 32) -> tuple[int, int]:
    """Make an extreme video aspect ratio acceptable to Qwen without dropping it."""

    def ceil_factor(value: int) -> int:
        return ((value + factor - 1) // factor) * factor

    target_height = ceil_factor(max(height, factor))
    target_width = ceil_factor(max(width, factor))
    # qwen-vl-utils rejects ratios above 200; use 199 to avoid a rounding edge.
    if target_height > 199 * target_width:
        target_width = ceil_factor((target_height + 198) // 199)
    elif target_width > 199 * target_height:
        target_height = ceil_factor((target_width + 198) // 199)
    return target_height, target_width


def _process_image_with_truncated_fallback(image: dict | str):
    """Retry otherwise valid truncated images without hiding unrelated failures."""
    from verl.utils.dataset.vision_utils import process_image

    try:
        return process_image(image)
    except OSError as error:
        message = str(error).casefold()
        if "broken data stream" not in message and "image file is truncated" not in message:
            raise

    from PIL import ImageFile

    path = image if isinstance(image, str) else image.get("image") or image.get("path")
    logger.warning("loading truncated image %s", path)
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return process_image(image)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _skipped_item(kind: str, entry: dict | str, source: str, error: Exception) -> dict:
    primary_key = {"image": "image", "video": "video", "signal": "signal"}[kind]
    path = entry if isinstance(entry, str) else str(entry.get(primary_key) or entry.get("path") or "<unknown>")
    logger.warning(
        "skipping unreadable %s source=%s path=%s (%s: %s)",
        kind,
        source,
        path,
        type(error).__name__,
        str(error)[:240],
    )
    return {
        "kind": "skipped",
        "failed_kind": kind,
        "path": path,
    }


def _process_video_with_extreme_aspect_fallback(video: dict, max_frames: int):
    """Process a video, resizing only when Qwen rejects its raw aspect ratio."""
    from verl.utils.dataset.vision_utils import process_video

    kwargs = {
        "image_patch_size": 16,
        "return_video_metadata": True,
        "nframes": max_frames,
    }
    try:
        return process_video(video, **kwargs)
    except ValueError as error:
        if "absolute aspect ratio must be smaller" not in str(error):
            raise

    from torchcodec.decoders import VideoDecoder

    path = video["video"]
    frame = VideoDecoder(path, num_ffmpeg_threads=1)[0]
    height, width = (int(value) for value in frame.shape[-2:])
    resized_height, resized_width = _factor_aligned_video_size(height, width)
    adjusted = dict(video)
    adjusted.update(resized_height=resized_height, resized_width=resized_width)
    logger.warning(
        "resizing extreme-aspect video %s from %dx%d to %dx%d",
        path,
        height,
        width,
        resized_height,
        resized_width,
    )
    return process_video(adjusted, **kwargs)


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

        tactile_rows = [
            row for row in rows if (signals := row.get("signals") or []) and _signal_family(signals[0]) == "tactile"
        ]
        if tactile_rows:
            stems = [_recording_stem(row) for row in tactile_rows]
            annotations = _annotation_targets(rows, set(stems))
            for row, stem in zip(tactile_rows, stems, strict=True):
                row["_tactile_targets"] = annotations[stem]
            for task in TASK_LABELS:
                observed = sum(task in row["_tactile_targets"] for row in tactile_rows)
                logger.info(
                    "tactile labels: task=%s observed=%d missing=%d",
                    task,
                    observed,
                    len(tactile_rows) - observed,
                )

        rows, visual_paths, repeated = _expand_and_deduplicate_visual_rows(rows)
        if visual_paths:
            logger.info(
                "kept %d unique image/video paths; removed %d repeated media entries",
                visual_paths,
                repeated,
            )

        self.rows = rows
        # Tactile rows are supervised by their joined QA spans, never by a vocabulary.
        vocab_sets: dict[str, set[str]] = {family: set() for family in _CLASSIFICATION_FAMILIES}
        self.sampling_groups: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(rows):
            signals = row.get("signals") or []
            if signals:
                kind = "signal"
                if (family := _signal_family(signals[0])) in vocab_sets:
                    vocab_sets[family].add(row["reward_model"]["ground_truth"])
            else:
                kind = "image" if row.get("images") else "video"
            self.sampling_groups.setdefault((kind, row["data_source"]), []).append(index)
        self.ts_label_vocabs = {family: tuple(sorted(labels)) for family, labels in vocab_sets.items() if labels}
        group_sizes = {f"{kind}/{source}": len(indices) for (kind, source), indices in self.sampling_groups.items()}
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
            try:
                media, family = self._load_signal(signals[0])
            except _SKIPPABLE_LOAD_ERRORS as error:
                return _skipped_item("signal", signals[0], sample["data_source"], error)
            return {
                "kind": "signal",
                "media": media,
                "family": family,
                "text": sample["reward_model"]["ground_truth"],
                **({"targets": sample["_tactile_targets"]} if "_tactile_targets" in sample else {}),
            }

        images = sample.get("images") or []
        if images:
            try:
                media = _process_image_with_truncated_fallback(images[0])
            except _SKIPPABLE_LOAD_ERRORS as error:
                return _skipped_item("image", images[0], sample["data_source"], error)
            return {"kind": "image", "media": media}

        video = sample["videos"][0]
        try:
            media = _process_video_with_extreme_aspect_fallback(video, self.max_video_frames)
        except _SKIPPABLE_LOAD_ERRORS as error:
            return _skipped_item("video", video, sample["data_source"], error)
        return {
            "kind": "video",
            "media": media,
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
                raise ValueError(f"signal_repeat_factors values must be positive integers; got {source}={factor!r}")

        effective_sizes = {group: len(rows) * self._repeat_factor(group) for group, rows in self.groups.items()}
        global_batch_size = self.batch_size * self.world_size
        batch_counts = {
            group: (size + global_batch_size - 1) // global_batch_size for group, size in effective_sizes.items()
        }
        self.num_batches = sum(
            count for group, count in batch_counts.items() if effective_sizes[group] >= self.world_size
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
    """Keep variable-shape media as one source-homogeneous list."""
    skipped = [item for item in batch if item["kind"] == "skipped"]
    valid = [item for item in batch if item["kind"] != "skipped"]
    if not valid:
        paths = ", ".join(item["path"] for item in skipped[:3])
        raise RuntimeError(f"all {len(batch)} rank-local samples failed to load; refusing an empty DDP batch: {paths}")
    kind = valid[0]["kind"]
    skipped_counts = {
        failed_kind: sum(item["failed_kind"] == failed_kind for item in skipped)
        for failed_kind in ("image", "video", "signal")
    }
    collated = {
        "kind": kind,
        "media": [item["media"] for item in valid],
        "family": valid[0].get("family"),
        "text": [item["text"] for item in valid] if kind == "signal" else [],
        "skipped": skipped_counts,
    }
    if kind == "signal" and "targets" in valid[0]:
        # One (B, 30) target and one (B, 30) weight over the concatenated task

        width = max(stop for _, stop in TACTILE_SPANS.values())
        targets = torch.zeros(len(valid), width, dtype=torch.float32)
        masks = torch.zeros(len(valid), width, dtype=torch.float32)
        for row, item in enumerate(valid):
            for task, (start, stop) in TACTILE_SPANS.items():
                masks[row, start:stop] = float(task in item["targets"])
                targets[row, [start + index for index in item["targets"].get(task, ())]] = 1.0
        collated["targets"] = targets
        collated["masks"] = masks
    return collated
