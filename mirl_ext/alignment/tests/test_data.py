# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mirl_ext.alignment.data import AlignmentDataset, HomogeneousBatchSampler, collate_alignment


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


def test_tactile_uses_complete_ground_truth_instead_of_filename_stem(tmp_path):
    captions = (
        "The participant quickly lifts the teapot while maintaining a stable grasp.",
        "The mug slips during the lift.",
    )
    rows = [
        {
            "data_source": "haptic_tactile",
            "signals": [{"signal": f"missing-{index}.pt", "format": "tactile_pt"}],
            "reward_model": {"ground_truth": caption},
            "extra_info": '{"stem":"2025-10-09_recognition_lift_idx0"}',
        }
        for index, caption in enumerate(captions)
    ]
    dataset = _dataset(tmp_path, "haptic", rows)

    assert dataset.ts_label_vocabs == {
        "tactile": (
            "The mug slips during the lift.",
            "The participant quickly lifts the teapot while maintaining a stable grasp.",
        ),
    }


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
    assert all(
        len(_batch_groups(dataset, batch0 + batch1)) == 1
        for batch0, batch1 in zip(rank0, rank1)
    )
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
    owner = {
        index: group
        for group, indices in dataset.sampling_groups.items()
        for index in indices
    }
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
