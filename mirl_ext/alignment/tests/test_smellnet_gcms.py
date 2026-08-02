# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import numpy as np
import torch

from mirl_ext.alignment.metrics import _smellnet_gcms_top1_metrics
from mirl_ext.alignment.objective import _paired_prototype_siglip_loss
from mirl_ext.alignment.smellnet_gcms import load_smellnet_gcms


def _write_bank(path) -> None:
    np.savez(
        path,
        food_labels=np.asarray(["Apple", "Pear"], dtype=object),
        vectors=np.asarray([[1.0, 10.0, 5.0], [3.0, 20.0, 5.0]], dtype=np.float32),
    )


def test_gcms_loader_matches_reorders_and_standardizes_exact_labels(tmp_path):
    path = tmp_path / "gcms.npz"
    _write_bank(path)

    bank = load_smellnet_gcms(path, ("pear", "apple"))

    assert bank.labels == ("pear", "apple")
    assert bank.features.shape == (2, 3)
    assert torch.allclose(bank.features[:, :2], torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
    assert torch.equal(bank.features[:, 2], torch.zeros(2))


def test_three_view_losses_and_top1_use_the_same_class_order():
    labels = ("apple", "pear", "banana")
    text = torch.eye(3, requires_grad=True)
    gcms = torch.eye(3, requires_grad=True)
    scale = torch.tensor(np.log(1.0 / 0.07), requires_grad=True)

    loss = _paired_prototype_siglip_loss((labels, gcms), (labels, text), scale)
    loss.backward()

    assert torch.isfinite(loss)
    assert gcms.grad is not None and text.grad is not None and scale.grad is not None
    metrics = _smellnet_gcms_top1_metrics(
        torch.eye(3),
        list(labels),
        ["smell"] * 3,
        (labels, torch.eye(3)),
        (labels, torch.eye(3)),
    )
    assert metrics == {
        "accuracy/smell_sensor_to_gcms": 1.0,
        "accuracy/smell_gcms_to_text": 1.0,
    }
