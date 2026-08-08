# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch
from omegaconf import OmegaConf

from mirl_ext.alignment.metrics import task_prediction_metrics
from mirl_ext.alignment.objective import (
    build_text_label_bank,
    compute_losses,
    task_siglip_loss,
)


def test_multi_positive_siglip_loss_is_finite():
    anchors = torch.tensor([[2**-0.5, 2**-0.5]])
    captions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    loss = task_siglip_loss(
        anchors,
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.tensor([True]),
        captions,
        bias=0.0,
        log_logit_scale=torch.tensor(math.log(10.0)),
    )
    assert loss is not None and torch.isfinite(loss)


def test_compute_losses_equally_averages_observed_tasks():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_logit_scale = torch.nn.Parameter(torch.tensor(math.log(2.0)))

        def forward(self, media):
            return torch.eye(2), self.log_logit_scale

    model = Model()
    bank = {
        "force_level": (("a", "b"), torch.eye(2), 0.0),
        "grip_stability": (("a", "b"), torch.flip(torch.eye(2), dims=(0,)), 0.0),
    }
    batch = {
        "media": [None, None],
        "targets": {
            "force_level": torch.eye(2),
            "grip_stability": torch.eye(2),
        },
        "masks": {
            "force_level": torch.tensor([True, True]),
            "grip_stability": torch.tensor([True, True]),
        },
    }
    total, metrics, _ = compute_losses(
        model,
        batch,
        OmegaConf.create({"loss": {"siglip_weight": 1.0}}),
        label_bank=bank,
    )
    expected = (metrics["loss/task/force_level"] + metrics["loss/task/grip_stability"]) / 2
    assert total.detach().item() == pytest.approx(expected)


def test_structured_metrics_support_multilabel_and_exclusive_tasks():
    scale = torch.tensor(math.log(10.0))
    multi_z = torch.tensor([[2**-0.5, 2**-0.5]])
    multi_text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    multi = task_prediction_metrics(
        multi_z,
        {"initial_fingers": torch.tensor([[1.0, 1.0, 0.0]])},
        {"initial_fingers": torch.tensor([True])},
        {"initial_fingers": (("thumb", "index", "palm"), multi_text, 0.0)},
        scale,
    )
    assert multi["f1_macro/task/initial_fingers"] == 1.0
    assert multi["recall_at_1/task/initial_fingers"] == 1.0
    assert multi["map/task/initial_fingers"] == 1.0

    exclusive = task_prediction_metrics(
        torch.eye(2),
        {"force_level": torch.eye(2)},
        {"force_level": torch.tensor([True, True])},
        {"force_level": (("light", "firm"), torch.eye(2), 0.0)},
        scale,
    )
    assert exclusive["accuracy/task/force_level"] == 1.0
    assert exclusive["f1_macro/task/force_level"] == 1.0
    assert exclusive["f1_macro/tactile"] == 1.0


def test_label_bank_bias_comes_from_train_positive_rate():
    class TextModel:
        def encode_text(self, labels, device):
            return torch.eye(len(labels))

    bank = build_text_label_bank(
        TextModel(),
        {"force_level": ("a", "b", "c", "d")},
        {"force_level": 0.25},
        torch.device("cpu"),
    )
    assert bank["force_level"][2] == pytest.approx(-math.log(3))
