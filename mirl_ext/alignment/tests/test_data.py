# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import ImageFile

from mirl_ext.alignment.data import (
    TACTILE_NUM_LABELS,
    TACTILE_SPANS,
    AlignmentDataset,
    HomogeneousBatchSampler,
    _annotation_targets,
    _factor_aligned_video_size,
    _process_image_with_truncated_fallback,
    collate_alignment,
)


def _dataset(tmp_path, name, rows, **kwargs):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return AlignmentDataset([str(path)], **kwargs)


def test_smellnet_mixture_never_enters_rows_or_label_vocab(tmp_path):
    rows = [
        {
            "data_source": source,
            "signals": [{"signal": f"missing-{index}.csv", "format": ""}],
            "reward_model": {"ground_truth": label},
        }
        for index, (source, label) in enumerate(
            (
                ("smellnet_base", "apple"),
                ("smellnet_base", "pear"),
                ("smellnet_mixture", "apple + pear"),
            )
        )
    ]
    dataset = _dataset(tmp_path, "smellnet", rows)

    assert len(dataset) == 2
    assert {row["data_source"] for row in dataset.rows} == {"smellnet_base"}
    assert dataset.ts_label_vocabs == {"smellnet": ("apple", "pear")}


def _haptic_signal_row(tensor_path, stem, caption="unused open description"):
    return {
        "data_source": "haptic_tactile",
        "signals": [
            {
                "signal": str(tensor_path),
                "format": "tactile_pt",
                "key": "right",
            }
        ],
        "reward_model": {"ground_truth": caption},
        "extra_info": json.dumps({"stem": stem}),
    }


def _tactile_annotation(stem, task, answer):
    return {
        "data_source": task,
        "videos": [{"video": f"/not-loaded/{stem}.mp4"}],
        "reward_model": {"ground_truth": answer},
        "extra_info": json.dumps({"video_path": f"visual-tactile/{stem}.mp4"}),
    }


def test_structured_tactile_join_changes_only_haptic_targets(tmp_path):
    tensor_path = tmp_path / "recording.pt"
    tactile = torch.randn(47, 16, 16)
    torch.save({"tactile": {"right": tactile}}, tensor_path)
    data_path = tmp_path / "signals.parquet"
    annotation_path = tmp_path / "annotations.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _haptic_signal_row(tensor_path, "recording"),
                {
                    "data_source": "smellnet_base",
                    "signals": [{"signal": "/not-loaded/apple.csv", "format": ""}],
                    "reward_model": {"ground_truth": "apple"},
                    "extra_info": None,
                },
            ]
        ),
        data_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _tactile_annotation("recording", "initial_fingers", "A,B,F"),
                _tactile_annotation("recording", "highest_pressure", "B,F"),
                _tactile_annotation("recording", "force_level", "B"),
                _tactile_annotation("recording", "grip_stability", "A"),
                _tactile_annotation("recording", "contact_feature", "C"),
                _tactile_annotation("recording", "local_shape", "B"),
                _tactile_annotation("recording", "description", "ignored free text"),
            ]
        ),
        annotation_path,
    )

    dataset = AlignmentDataset(
        [str(data_path), str(annotation_path)],
    )
    sample = dataset[0]

    assert torch.equal(sample["media"], tactile)
    assert sample["targets"]["initial_fingers"] == (0, 1, 5)
    assert sample["targets"]["force_level"] == (1,)
    assert dataset.ts_label_vocabs == {"smellnet": ("apple",)}
    assert set(dataset.sampling_groups) == {
        ("signal", "haptic_tactile"),
        ("signal", "smellnet_base"),
        ("video", "initial_fingers"),
    }

    batch = collate_alignment([sample])
    # One (1, 30) target/mask pair over the concatenated task vocabularies; a task's
    # answer lands inside its own column span.
    assert batch["targets"].shape == (1, TACTILE_NUM_LABELS)
    assert batch["masks"].shape == (1, TACTILE_NUM_LABELS)
    start, stop = TACTILE_SPANS["initial_fingers"]
    assert torch.equal(
        batch["targets"][0, start:stop],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    # Every task this fixture answers is fully unmasked; a task's columns are all-or-
    # nothing, which is what lets the metrics recover the per-row mask from column 0.
    for task, (task_start, task_stop) in TACTILE_SPANS.items():
        expected = 1.0 if task in sample["targets"] else 0.0
        assert batch["masks"][0, task_start:task_stop].tolist() == [expected] * (task_stop - task_start), task
    # Exactly the answered positives are set, and nothing outside them.
    assert batch["targets"].sum() == sum(len(v) for v in sample["targets"].values())


def test_structured_tactile_join_masks_conflicting_answers():
    rows = [
        _tactile_annotation("recording", "force_level", "A"),
        _tactile_annotation("recording", "force_level", "B"),
    ]
    assert _annotation_targets(rows, {"recording"})["recording"] == {}


def test_visual_annotations_are_deduplicated_by_media_path(tmp_path):
    rows = [
        {
            "data_source": "tactile",
            "images": None,
            "videos": [{"video": "/data/shared.mp4"}],
            "reward_model": {"ground_truth": answer},
        }
        for answer in ("A", "B", "C")
    ]
    rows += [
        {
            "data_source": "climb",
            "images": [{"image": "/data/shared.png"}],
            "videos": None,
            "reward_model": {"ground_truth": answer},
        }
        for answer in ("normal", "abnormal")
    ]

    dataset = _dataset(tmp_path, "visual", rows)

    assert len(dataset) == 2
    assert len(dataset.sampling_groups[("image", "climb")]) == 1
    assert len(dataset.sampling_groups[("video", "tactile")]) == 1
    assert all("reward_model" not in row and "data_source" in row for row in dataset.rows)


def test_multi_media_rows_expand_every_unique_path(tmp_path):
    rows = [
        {
            "data_source": "climb",
            "images": [{"image": f"/data/image-{index}.png"} for index in range(4)],
            "videos": None,
            "reward_model": {"ground_truth": "first annotation"},
        },
        {
            "data_source": "climb",
            "images": [
                {"image": "/data/image-0.png"},
                {"image": "/data/image-4.png"},
            ],
            "videos": None,
            "reward_model": {"ground_truth": "repeated annotation"},
        },
        {
            "data_source": "tactile",
            "images": None,
            "videos": [
                {"video": "/data/video-0.mp4"},
                {"video": "/data/video-1.mp4"},
            ],
            "reward_model": {"ground_truth": "video annotation"},
        },
    ]

    dataset = _dataset(tmp_path, "multi-visual", rows)

    assert len(dataset) == 7
    assert len(dataset.sampling_groups[("image", "climb")]) == 5
    assert len(dataset.sampling_groups[("video", "tactile")]) == 2
    assert all(len(row.get("images") or []) + len(row.get("videos") or []) == 1 for row in dataset.rows)
    assert {entry[0]["image"] for row in dataset.rows if (entry := row.get("images"))} == {
        f"/data/image-{index}.png" for index in range(5)
    }
    assert {entry[0]["video"] for row in dataset.rows if (entry := row.get("videos"))} == {
        "/data/video-0.mp4",
        "/data/video-1.mp4",
    }

    rank0 = list(HomogeneousBatchSampler(dataset, 4, rank=0, world_size=2, seed=5))
    rank1 = list(HomogeneousBatchSampler(dataset, 4, rank=1, world_size=2, seed=5))
    visited = [index for batches in zip(rank0, rank1, strict=True) for batch in batches for index in batch]
    assert len(visited) == len(set(visited)) == len(dataset)


def test_extreme_video_size_is_retained_with_safe_factor_alignment():
    assert _factor_aligned_video_size(512, 2) == (512, 32)
    assert _factor_aligned_video_size(2, 512) == (32, 512)
    assert _factor_aligned_video_size(480, 640) == (480, 640)


def test_truncated_image_retry_is_scoped(monkeypatch, caplog):
    calls = []

    def process_image(image):
        calls.append((image, ImageFile.LOAD_TRUNCATED_IMAGES))
        if not ImageFile.LOAD_TRUNCATED_IMAGES:
            raise OSError("broken data stream when reading image file")
        return "decoded"

    monkeypatch.setattr("verl.utils.dataset.vision_utils.process_image", process_image)
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)
    image = {"image": "/data/truncated.jpg"}

    assert _process_image_with_truncated_fallback(image) == "decoded"
    assert calls == [(image, False), (image, True)]
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    assert "/data/truncated.jpg" in caplog.text


def test_image_retry_does_not_hide_unrelated_os_errors(monkeypatch):
    def process_image(_image):
        raise OSError("permission denied")

    monkeypatch.setattr("verl.utils.dataset.vision_utils.process_image", process_image)

    with pytest.raises(OSError, match="permission denied"):
        _process_image_with_truncated_fallback({"image": "/data/unreadable.jpg"})


def test_dataset_marks_unreadable_image_for_logged_skip(tmp_path, monkeypatch, caplog):
    dataset = _dataset(
        tmp_path,
        "unreadable",
        [
            {
                "data_source": "chest_xray",
                "images": [{"image": "/data/unreadable.jpg"}],
                "videos": None,
                "signals": None,
            }
        ],
    )

    def process_image(_image):
        raise OSError("decoder rejected image")

    monkeypatch.setattr("mirl_ext.alignment.data._process_image_with_truncated_fallback", process_image)

    assert dataset[0] == {
        "kind": "skipped",
        "failed_kind": "image",
        "path": "/data/unreadable.jpg",
    }
    assert "source=chest_xray" in caplog.text


def test_collate_filters_failed_items_and_counts_them():
    batch = collate_alignment(
        [
            {"kind": "image", "media": "valid-a"},
            {"kind": "skipped", "failed_kind": "image", "path": "/bad-a.jpg"},
            {"kind": "image", "media": "valid-b"},
        ]
    )

    assert batch == {
        "kind": "image",
        "media": ["valid-a", "valid-b"],
        "family": None,
        "text": [],
        "skipped": {"image": 1, "video": 0, "signal": 0},
    }


def test_collate_rejects_an_entirely_unreadable_local_batch():
    with pytest.raises(RuntimeError, match="refusing an empty DDP batch"):
        collate_alignment([{"kind": "skipped", "failed_kind": "video", "path": "/bad.mp4"}])


def test_collate_keeps_complete_source_homogeneous_signals():
    signals = [
        torch.arange(12, dtype=torch.float32).reshape(2, 6),
        torch.arange(16, dtype=torch.float32).reshape(2, 8),
    ]
    batch = collate_alignment(
        [
            {
                "kind": "signal",
                "media": signal,
                "family": "smellnet",
                "text": label,
            }
            for signal, label in zip(signals, ("apple", "pear"), strict=True)
        ]
    )

    assert batch["kind"] == "signal"
    assert batch["family"] == "smellnet"
    assert all(actual is expected for actual, expected in zip(batch["media"], signals, strict=True))
    assert batch["text"] == ["apple", "pear"]
    assert batch["skipped"] == {"image": 0, "video": 0, "signal": 0}


class _GroupedDataset:
    def __init__(self):
        self.sampling_groups = {
            ("signal", "smellnet_base"): list(range(0, 12)),
            ("signal", "ecg"): list(range(12, 24)),
            ("signal", "haptic_tactile"): list(range(24, 36)),
            ("image", "climb"): list(range(36, 48)),
            ("video", "human_behaviour"): list(range(48, 60)),
        }


def _group_counts(dataset, batch):
    owner = {index: group for group, indices in dataset.sampling_groups.items() for index in indices}
    return {group: sum(owner[index] == group for index in batch) for group in dataset.sampling_groups}


def _batch_groups(dataset, batch):
    counts = _group_counts(dataset, batch)
    return {group for group, count in counts.items() if count}


def test_homogeneous_sampler_consumes_all_rows_once_across_ranks():
    dataset = _GroupedDataset()
    rank0 = HomogeneousBatchSampler(
        dataset,
        batch_size=8,
        rank=0,
        world_size=2,
        seed=7,
    )
    rank1 = HomogeneousBatchSampler(
        dataset,
        batch_size=8,
        rank=1,
        world_size=2,
        seed=7,
    )

    seen = []
    for batch0, batch1 in zip(rank0, rank1, strict=True):
        assert len(batch0) <= 8 and len(batch1) <= 8
        assert len(_batch_groups(dataset, batch0 + batch1)) == 1
        assert set(batch0).isdisjoint(batch1)
        seen.extend(batch0)
        seen.extend(batch1)

    assert len(seen) == len(set(seen))
    assert set(seen) == set(range(60))


def test_distributed_sampler_keeps_tail_ranks_in_lockstep():
    dataset = _GroupedDataset()
    dataset.sampling_groups = {
        ("signal", "smellnet_base"): list(range(5)),
        ("signal", "ecg"): list(range(5, 9)),
        ("signal", "haptic_tactile"): list(range(9, 13)),
    }
    rank0 = list(HomogeneousBatchSampler(dataset, 6, rank=0, world_size=2, seed=3))
    rank1 = list(HomogeneousBatchSampler(dataset, 6, rank=1, world_size=2, seed=3))

    assert len(rank0) == len(rank1) == 3
    assert all(batch for batch in rank0 + rank1)
    assert all(len(_batch_groups(dataset, batch0 + batch1)) == 1 for batch0, batch1 in zip(rank0, rank1, strict=False))
    assert set(index for batch in rank0 + rank1 for index in batch) == set(range(13))


def test_homogeneous_sampler_is_deterministic_per_epoch():
    dataset = _GroupedDataset()
    sampler = HomogeneousBatchSampler(dataset, 8, seed=11)
    first = list(sampler)
    assert first == list(HomogeneousBatchSampler(dataset, 8, seed=11))
    flattened = [index for batch in first for index in batch]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(range(60))

    sampler.set_epoch(1)
    assert first != list(sampler)


def test_homogeneous_sampler_repeats_only_configured_signal_sources():
    dataset = _GroupedDataset()
    rank0 = HomogeneousBatchSampler(
        dataset,
        batch_size=8,
        rank=0,
        world_size=2,
        seed=13,
        signal_repeat_factors={"smellnet_base": 3, "haptic_tactile": 2},
    )
    rank1 = HomogeneousBatchSampler(
        dataset,
        batch_size=8,
        rank=1,
        world_size=2,
        seed=13,
        signal_repeat_factors={"smellnet_base": 3, "haptic_tactile": 2},
    )

    counts = {group: 0 for group in dataset.sampling_groups}
    owner = {index: group for group, indices in dataset.sampling_groups.items() for index in indices}
    for batch0, batch1 in zip(rank0, rank1, strict=True):
        assert len(_batch_groups(dataset, batch0 + batch1)) == 1
        for index in batch0 + batch1:
            counts[owner[index]] += 1

    assert counts == {
        ("signal", "smellnet_base"): 36,
        ("signal", "ecg"): 12,
        ("signal", "haptic_tactile"): 24,
        ("image", "climb"): 12,
        ("video", "human_behaviour"): 12,
    }


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_homogeneous_sampler_rejects_non_positive_integer_repeat_factors(factor):
    with pytest.raises(ValueError, match="positive integers"):
        HomogeneousBatchSampler(
            _GroupedDataset(),
            batch_size=8,
            signal_repeat_factors={"smellnet_base": factor},
        )


def test_homogeneous_sampler_rejects_unknown_repeat_sources():
    with pytest.raises(ValueError, match="unknown signal sources"):
        HomogeneousBatchSampler(
            _GroupedDataset(),
            batch_size=8,
            signal_repeat_factors={"missing": 2},
        )
