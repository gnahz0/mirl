# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import json

from mirl_ext.alignment.data import AlignmentDataset, FamilyBalancedBatchSampler


def test_fixed_label_vocab_is_built_before_dataset_sampling(tmp_path):
    rows = [
        {
            "data_source": "ecg",
            "signals": [{"signal": f"missing-{label}.pt", "format": "ts_pt"}],
            "reward_model": {"ground_truth": label},
        }
        for label in ("Normal", "Hypertrophy", "Other")
    ]
    path = tmp_path / "ecg.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = AlignmentDataset(
        str(path),
        max_samples=1,
        seed=0,
        enable_videos=False,
    )

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
    path = tmp_path / "ecg.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = AlignmentDataset(str(path), enable_videos=False)

    assert dataset.ts_label_vocabs == {"ecg": ("normal", "other")}


def test_excluded_source_never_enters_rows_or_label_vocab(tmp_path):
    rows = [
        {
            "data_source": source,
            "signals": [{"signal": f"missing-{index}.csv"}],
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
    path = tmp_path / "smellnet.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = AlignmentDataset(
        str(path),
        exclude_data_sources=["smellnet_mixture"],
        enable_videos=False,
    )

    assert len(dataset) == 2
    assert {row["data_source"] for row in dataset.rows} == {"smellnet_base"}
    assert dataset.ts_label_vocabs == {"smell": ("apple", "pear")}


def test_haptic_stem_task_labels_remove_participants_and_replicates(tmp_path):
    stems = (
        "2025-10-09_simin_recognition_lift_fast_teapotD_idx0",
        "2025-09-04_recognition_lift_fast_mugC_rao_idx0",
        "2026-01-03_AAA_insert_USB_sideways",
        "2025-05-04-hit_idx0",
        "2025-09-14_jiayi_idx0",
    )
    rows = [
        {
            "data_source": "haptic_tactile",
            "signals": [{"signal": f"missing-{index}.pt", "format": "tactile_pt"}],
            "reward_model": {"ground_truth": f"unique caption {index}"},
            "extra_info": json.dumps({"stem": stem}),
        }
        for index, stem in enumerate(stems)
    ]
    path = tmp_path / "haptic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = AlignmentDataset(
        str(path),
        tactile_label_mode="stem_task_pair",
        enable_videos=False,
    )

    assert dataset.ts_label_vocabs == {
        "tactile": (
            "hit",
            "insert usb",
            "recognition lift",
            "unclassified haptic task",
        ),
    }


class _GroupedDataset:
    def __init__(self):
        self.sampling_groups = {
            "smell": list(range(0, 12)),
            "ecg": list(range(12, 24)),
            "tactile": list(range(24, 36)),
            "img": list(range(36, 60)),
        }

    def __len__(self):
        return 60


def _group_counts(dataset, batch):
    owner = {index: family for family, indices in dataset.sampling_groups.items() for index in indices}
    return {family: sum(owner[index] == family for index in batch) for family in dataset.sampling_groups}


def test_family_balanced_sampler_has_exact_disjoint_rank_quotas():
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

    for batch0, batch1 in zip(rank0, rank1, strict=True):
        assert _group_counts(dataset, batch0) == {
            "smell": 2,
            "ecg": 2,
            "tactile": 2,
            "img": 2,
        }
        assert _group_counts(dataset, batch1) == {
            "smell": 2,
            "ecg": 2,
            "tactile": 2,
            "img": 2,
        }
        assert set(batch0).isdisjoint(batch1)


def test_family_balanced_sampler_is_deterministic_per_epoch():
    dataset = _GroupedDataset()
    sampler = FamilyBalancedBatchSampler(dataset, 8, 2, seed=11)
    first = list(sampler)
    assert first == list(FamilyBalancedBatchSampler(dataset, 8, 2, seed=11))
    assert {index for batch in first for index in batch} == set(range(len(dataset)))

    sampler.set_epoch(1)
    assert first != list(sampler)
