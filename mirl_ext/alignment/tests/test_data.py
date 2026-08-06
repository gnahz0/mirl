# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mirl_ext.alignment.data import AlignmentDataset, FamilyBalancedBatchSampler, collate_alignment


def _dataset(tmp_path, name, rows, **kwargs):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return AlignmentDataset([str(path)], **kwargs)


def test_fixed_label_vocab_is_built_before_dataset_sampling(tmp_path):
    rows = [
        {
            "data_source": "ecg",
            "signals": [{"signal": f"missing-{label}.pt", "format": "ts_pt"}],
            "reward_model": {"ground_truth": label},
        }
        for label in ("Normal", "Hypertrophy", "Other")
    ]
    dataset = _dataset(tmp_path, "ecg", rows, max_samples=1, seed=0)

    assert len(dataset) == 1
    assert dataset.ts_label_vocabs == {
        "ecg": ("hypertrophy", "normal", "other"),
    }


def test_label_identity_is_casefolded_and_whitespace_normalized(tmp_path):
    rows = [
        {
            "data_source": "ecg",
            "signals": [{"signal": f"missing-{index}.pt", "format": "ts_pt"}],
            "reward_model": {"ground_truth": label},
        }
        for index, label in enumerate(("Normal", " normal ", "NORMAL", "Other"))
    ]
    dataset = _dataset(tmp_path, "ecg", rows)

    assert dataset.ts_label_vocabs == {"ecg": ("normal", "other")}


def test_smellnet_mixture_never_enters_rows_or_label_vocab(tmp_path):
    rows = [
        {
            "data_source": source,
            "signals": [{"signal": f"missing-{index}.csv", "format": ""}],
            "reward_model": {"ground_truth": label},
        }
        for index, (source, label) in enumerate(
            (
                ("smellnet_base", "Apple"),
                ("smellnet_base", "Pear"),
                ("smellnet_mixture", "Apple + Pear"),
            )
        )
    ]
    dataset = _dataset(tmp_path, "smellnet", rows)

    assert len(dataset) == 2
    assert {row["data_source"] for row in dataset.rows} == {"smellnet_base"}
    assert dataset.ts_label_vocabs == {"smell": ("apple", "pear")}


def test_tactile_uses_complete_ground_truth_instead_of_filename_stem(tmp_path):
    captions = (
        "The participant quickly lifts the teapot while maintaining a stable grasp.",
        "  The mug slips during the lift.  ",
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
            "the mug slips during the lift.",
            "the participant quickly lifts the teapot while maintaining a stable grasp.",
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
    assert len(dataset.sampling_groups["img"]) == 2
    assert all("reward_model" not in row for row in dataset.rows)


def test_collate_keeps_complete_native_signals():
    signal = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    tactile = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 2, 2)
    force = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)
    batch = collate_alignment(
        [
            {
                "kind": "signal",
                "media": signal,
                "family": "smell",
                "text": "apple",
            },
            {
                "kind": "signal",
                "media": {"tactile": tactile, "force": force},
                "family": "tactile",
                "text": "the grasp remains stable while lifting the mug",
            },
        ]
    )

    assert len(batch["ts_signal"]) == 2
    assert torch.equal(batch["ts_signal"][0], signal)
    assert torch.equal(batch["ts_signal"][1]["tactile"], tactile)
    assert torch.equal(batch["ts_signal"][1]["force"], force)
    assert batch["ts_format"] == ["smell", "tactile"]
    assert batch["ts_signal_text"] == ["apple", "the grasp remains stable while lifting the mug"]


class _GroupedDataset:
    def __init__(self):
        self.sampling_groups = {
            "smell": list(range(0, 12)),
            "ecg": list(range(12, 24)),
            "tactile": list(range(24, 36)),
            "img": list(range(36, 60)),
        }


def _group_counts(dataset, batch):
    owner = {index: family for family, indices in dataset.sampling_groups.items() for index in indices}
    return {family: sum(owner[index] == family for index in batch) for family in dataset.sampling_groups}


def test_family_balanced_sampler_consumes_all_rows_once_across_ranks():
    dataset = _GroupedDataset()
    rank0 = FamilyBalancedBatchSampler(
        dataset,
        batch_size=8,
        ts_per_family=2,
        rank=0,
        world_size=2,
        seed=7,
    )
    rank1 = FamilyBalancedBatchSampler(
        dataset,
        batch_size=8,
        ts_per_family=2,
        rank=1,
        world_size=2,
        seed=7,
    )

    seen = []
    for batch0, batch1 in zip(rank0, rank1, strict=True):
        assert len(batch0) <= 8 and len(batch1) <= 8
        for counts in (_group_counts(dataset, batch0), _group_counts(dataset, batch1)):
            assert all(counts[family] <= 2 for family in ("smell", "ecg", "tactile"))
        assert set(batch0).isdisjoint(batch1)
        seen.extend(batch0)
        seen.extend(batch1)

    assert len(seen) == len(set(seen))
    assert set(seen) == set(range(60))


def test_distributed_sampler_keeps_tail_ranks_in_lockstep():
    dataset = _GroupedDataset()
    dataset.sampling_groups = {
        "smell": list(range(5)),
        "ecg": list(range(5, 9)),
        "tactile": list(range(9, 13)),
        "img": [],
    }
    rank0 = list(FamilyBalancedBatchSampler(dataset, 6, 2, rank=0, world_size=2, seed=3))
    rank1 = list(FamilyBalancedBatchSampler(dataset, 6, 2, rank=1, world_size=2, seed=3))

    assert len(rank0) == len(rank1) == 2
    assert set(index for batch in rank0 + rank1 for index in batch) == set(range(13))


def test_family_balanced_sampler_is_deterministic_per_epoch():
    dataset = _GroupedDataset()
    sampler = FamilyBalancedBatchSampler(dataset, 8, 2, seed=11)
    first = list(sampler)
    assert first == list(FamilyBalancedBatchSampler(dataset, 8, 2, seed=11))
    flattened = [index for batch in first for index in batch]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(range(60))

    sampler.set_epoch(1)
    assert first != list(sampler)


def test_family_balanced_sampler_does_not_recycle_smaller_families():
    dataset = _GroupedDataset()
    dataset.sampling_groups["smell"] = dataset.sampling_groups["smell"][:4]
    sampler = FamilyBalancedBatchSampler(dataset, 8, 2, seed=5)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == 7
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set().union(*map(set, dataset.sampling_groups.values()))
    assert sum(index in dataset.sampling_groups["smell"] for index in flattened) == 4
    smell_batches = [i for i, batch in enumerate(batches) if set(batch) & set(dataset.sampling_groups["smell"])]
    assert smell_batches[-1] > smell_batches[0] + 1
