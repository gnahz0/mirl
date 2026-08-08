# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

import math

import pytest
import torch

from mirl_ext.alignment.metrics import _ts_prediction_metrics
from mirl_ext.alignment.objective import _label_siglip_loss


def test_tactile_siglip_uses_complete_caption_bank():
    embeddings = torch.eye(3)
    labels = ("lift", "slip", "stable")
    loss = _label_siglip_loss(
        embeddings[:2],
        ["lift", "slip"],
        labels,
        embeddings,
        torch.tensor(math.log(10.0)),
    )
    assert torch.isfinite(loss)


def test_tactile_retrieval_metrics():
    labels = ("lift", "slip", "stable", "rotate", "press", "release")
    text = torch.eye(6)
    tactile = torch.stack((text[0], text[5]))
    metrics = _ts_prediction_metrics(
        tactile,
        ["lift", "release"],
        ["tactile", "tactile"],
        {"tactile": (labels, text)},
    )
    assert metrics["recall_at_1/ts_tactile"] == 1.0
    assert metrics["recall_at_5/ts_tactile"] == 1.0
    assert metrics["map/ts_tactile"] == 1.0


def test_siglip_reduction_sums_candidates_then_averages_anchors():
    anchors = torch.eye(2)
    captions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scale = torch.tensor(math.log(2.0))
    loss = _label_siglip_loss(anchors, ["a", "b"], ("a", "b", "c"), captions, scale)
    logits = scale.exp() * anchors @ captions.T - math.log(2)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        reduction="none",
    ).sum(dim=1).mean()
    assert loss == pytest.approx(expected)
