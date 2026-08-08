# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mirl_ext.alignment.data import AlignmentDataset, HomogeneousBatchSampler, collate_alignment


def test_tactile_dataset_loads_full_recording_and_caption(tmp_path):
    tensor_path = tmp_path / "recording.pt"
    tactile = torch.randn(47, 16, 16)
    torch.save({"tactile": {"right": tactile}}, tensor_path)
    parquet_path = tmp_path / "tactile.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "data_source": "haptic_tactile",
                    "signals": [
                        {
                            "signal": str(tensor_path),
                            "format": "tactile_pt",
                            "key": "right",
                        }
                    ],
                    "reward_model": {"ground_truth": "The grasp remains stable."},
                }
            ]
        ),
        parquet_path,
    )

    dataset = AlignmentDataset([str(parquet_path)])
    sample = dataset[0]

    assert torch.equal(sample["media"], tactile)
    assert sample["text"] == "The grasp remains stable."
    assert dataset.ts_label_vocabs == {"tactile": ("The grasp remains stable.",)}
    batch = collate_alignment([sample])
    assert batch["family"] == "tactile" and batch["media"][0].shape == (47, 16, 16)


def test_sampler_repeats_only_complete_tactile_passes():
    dataset = type(
        "Dataset",
        (),
        {"sampling_groups": {("signal", "haptic_tactile"): list(range(12))}},
    )()
    sampler = HomogeneousBatchSampler(
        dataset,
        batch_size=4,
        signal_repeat_factors={"haptic_tactile": 2},
    )
    sampled = [index for batch in sampler for index in batch]
    assert sorted(sampled) == sorted(list(range(12)) * 2)
