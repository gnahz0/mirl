# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mirl_ext.alignment.data import AlignmentDataset, HomogeneousBatchSampler, collate_alignment


def _write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def _signal_row(tensor_path, stem):
    return {
        "data_source": "haptic_tactile",
        "signals": [{"signal": str(tensor_path), "format": "tactile_pt", "key": "right"}],
        "reward_model": {"style": "open", "ground_truth": "unused open description"},
        "extra_info": json.dumps({"stem": stem}),
    }


def _annotation(stem, task, answer):
    return {
        "data_source": task,
        "videos": [{"video": f"/not-loaded/{stem}.mp4"}],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": json.dumps({"video_path": f"visual-tactile/{stem}.mp4"}),
    }


def test_dataset_joins_closed_targets_and_ignores_open_responses(tmp_path):
    tensor_path = tmp_path / "recording.pt"
    tactile = torch.randn(47, 16, 16)
    torch.save({"tactile": {"right": tactile}}, tensor_path)
    data_path = tmp_path / "signals.parquet"
    annotation_path = tmp_path / "annotations.parquet"
    _write_parquet(data_path, [_signal_row(tensor_path, "recording")])
    _write_parquet(
        annotation_path,
        [
            _annotation("recording", "initial_fingers", "A,B,F"),
            _annotation("recording", "force_level", "B"),
            _annotation("recording", "description", "This text is never a target."),
        ],
    )

    dataset = AlignmentDataset(
        [str(data_path)],
        [str(annotation_path)],
        ["initial_fingers", "force_level"],
    )
    sample = dataset[0]

    assert torch.equal(sample["media"], tactile)
    assert sample["targets"] == {"initial_fingers": (0, 1, 5), "force_level": (1,)}
    assert dataset.task_positive_rates == {
        "initial_fingers": 0.5,
        "force_level": 0.25,
    }
    batch = collate_alignment([sample])
    assert torch.equal(
        batch["targets"]["initial_fingers"],
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    assert batch["masks"]["force_level"].tolist() == [True]


def test_dataset_deduplicates_matches_and_masks_conflicts(tmp_path):
    tensor_path = tmp_path / "recording.pt"
    other_tensor_path = tmp_path / "other.pt"
    torch.save({"tactile": {"right": torch.randn(8, 16, 16)}}, tensor_path)
    torch.save({"tactile": {"right": torch.randn(8, 16, 16)}}, other_tensor_path)
    data_path = tmp_path / "signals.parquet"
    annotation_path = tmp_path / "annotations.parquet"
    _write_parquet(
        data_path,
        [_signal_row(tensor_path, "recording"), _signal_row(other_tensor_path, "other")],
    )
    _write_parquet(
        annotation_path,
        [
            _annotation("recording", "initial_fingers", "A,F"),
            _annotation("recording", "initial_fingers", "A,F"),
            _annotation("recording", "force_level", "A"),
            _annotation("recording", "force_level", "B"),
            _annotation("other", "force_level", "C"),
        ],
    )

    dataset = AlignmentDataset(
        [str(data_path)],
        [str(annotation_path)],
        ["initial_fingers", "force_level"],
    )
    sample = dataset[0]
    batch = collate_alignment([sample])

    assert sample["targets"] == {"initial_fingers": (0, 5)}
    assert batch["masks"]["initial_fingers"].tolist() == [True]
    assert batch["masks"]["force_level"].tolist() == [False]


def test_sampler_uses_each_tactile_row_once():
    dataset = list(range(12))
    sampler = HomogeneousBatchSampler(dataset, batch_size=4)
    sampled = [index for batch in sampler for index in batch]
    assert sorted(sampled) == list(range(12))
