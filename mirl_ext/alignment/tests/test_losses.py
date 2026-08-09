# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from mirl_ext.alignment.metrics import (  # noqa: E402
    _allreduce_metrics,
    _label_ranking_metrics,
    _merge_prediction_metrics,
    _metric_groups,
    _tactile_prediction_metrics,
    _ts_prediction_metrics,
    add_batch_counts,
)
from mirl_ext.alignment.objective import (  # noqa: E402
    _build_tactile_label_bank,
    _compute_losses,
    _label_siglip_loss,
    _tactile_task_siglip_loss,
)


def test_label_loss_is_class_balanced():
    pa = torch.tensor([1.0, 0.0])
    pb = torch.tensor([0.0, 1.0])
    scale = torch.tensor(math.log(10.0))

    def compute(n_a: int):
        z = torch.stack([pa] * n_a + [pb])
        return _label_siglip_loss(
            z,
            ["A"] * n_a + ["B"],
            ("A", "B"),
            torch.stack([pa, pb]),
            scale,
        )

    assert float(compute(2)) == pytest.approx(float(compute(20)), rel=1e-6)


def test_label_loss_sums_candidates_before_averaging_anchors():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    candidate_labels = ("A", "B", "C")
    labels = ["A", "B"]
    log_scale = torch.tensor(math.log(2.0))

    loss = _label_siglip_loss(
        anchors,
        labels,
        candidate_labels,
        prototypes,
        log_scale,
    )

    bias = -math.log(len(candidate_labels) - 1)
    logits = log_scale.exp() * (anchors @ prototypes.T) + bias
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        .sum(dim=1)
        .mean()
    )
    assert loss == pytest.approx(expected)


def test_label_loss_uses_absent_labels_as_negatives():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scale = torch.tensor(math.log(10.0))
    loss = _label_siglip_loss(
        anchors,
        ["A", "B"],
        ("A", "B", "C"),
        prototypes,
        scale,
    )
    assert torch.isfinite(loss)


def test_label_bank_handles_unique_tactile_answers():
    captions = ["the cup slips during the lift", "the grasp remains stable"]
    loss = _label_siglip_loss(
        torch.eye(2),
        captions,
        tuple(captions),
        torch.eye(2),
        torch.tensor(math.log(10.0)),
    )

    assert torch.isfinite(loss)


def test_structured_tactile_loss_accepts_multiple_positive_labels():
    anchors = torch.tensor([[2**-0.5, 2**-0.5]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    loss = _tactile_task_siglip_loss(
        anchors,
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.tensor([True]),
        prototypes,
        bias=0.0,
        log_logit_scale=torch.tensor(math.log(10.0)),
    )

    assert loss is not None and torch.isfinite(loss)


def test_general_objective_averages_structured_tasks_only_for_tactile():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_logit_scale = torch.nn.Parameter(torch.tensor(math.log(2.0)))

        def forward(self, kind, media, family, max_image_tokens):
            assert (kind, family, max_image_tokens) == ("signal", "tactile", 1024)
            return None, None, None, torch.eye(2), self.log_logit_scale

    bank = {
        "force_level": (("a", "b"), torch.eye(2), 0.0),
        "grip_stability": (
            ("a", "b"),
            torch.flip(torch.eye(2), dims=(0,)),
            0.0,
        ),
    }
    batch = {
        "kind": "signal",
        "media": [None, None],
        "family": "tactile",
        "text": ["unused", "unused"],
        "targets": {
            "force_level": torch.eye(2),
            "grip_stability": torch.eye(2),
        },
        "masks": {
            "force_level": torch.tensor([True, True]),
            "grip_stability": torch.tensor([True, True]),
        },
    }
    cfg = type(
        "Cfg",
        (),
        {
            "data": type("Data", (), {"max_image_tokens": 1024})(),
            "loss": type("Loss", (), {"siglip_weight": 1.0, "distill_weight": 1.0})(),
        },
    )()

    total, metrics, task_eval = _compute_losses(
        Model(),
        batch,
        cfg,
        label_bank={},
        tactile_label_bank=bank,
    )

    expected = (metrics["loss/task/force_level"] + metrics["loss/task/grip_stability"]) / 2
    assert total.detach().item() == pytest.approx(expected)
    assert metrics["loss/ts_tactile"] == pytest.approx(expected)
    assert set(task_eval[3]) == set(bank)


def test_structured_tactile_metrics_merge_as_one_family():
    scale = torch.tensor(math.log(10.0))
    multi_z = torch.tensor([[2**-0.5, 2**-0.5]])
    multi_text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    tactile = _tactile_prediction_metrics(
        multi_z,
        {"initial_fingers": torch.tensor([[1.0, 1.0, 0.0]])},
        {"initial_fingers": torch.tensor([True])},
        {
            "initial_fingers": (
                ("thumb", "index", "palm"),
                multi_text,
                0.0,
            )
        },
        scale,
    )
    merged = _merge_prediction_metrics(
        {"f1_macro/ts_smellnet": 0.5, "f1_macro/overall": 0.5},
        tactile,
    )

    assert tactile["accuracy/task/initial_fingers"] == 1.0
    assert tactile["f1_macro/task/initial_fingers"] == 1.0
    assert tactile["recall_at_1/ts_tactile"] == 1.0
    assert tactile["map/ts_tactile"] == 1.0
    assert merged["f1_macro/overall"] == pytest.approx(0.75)


def test_structured_tactile_bank_bias_uses_train_positive_rate():
    class TextModel:
        def encode_text(self, labels, device):
            return torch.eye(len(labels))

    bank = _build_tactile_label_bank(
        TextModel(),
        {"force_level": ("a", "b", "c", "d")},
        {"force_level": 0.25},
        torch.device("cpu"),
    )

    assert bank["force_level"][2] == pytest.approx(-math.log(3))


def test_effective_batch_macro_f1_is_not_an_average_of_micro_batches():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    first = _label_ranking_metrics(
        torch.tensor([[1.0, 0.0]]),
        ["A"],
        ("A", "B"),
        prototypes,
    )
    second = _label_ranking_metrics(
        torch.tensor([[1.0, 0.0]]),
        ["B"],
        ("A", "B"),
        prototypes,
    )
    combined = _label_ranking_metrics(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ["A", "B"],
        ("A", "B"),
        prototypes,
    )

    assert (first["f1_macro"] + second["f1_macro"]) / 2 == pytest.approx(0.5)
    assert combined["f1_macro"] == pytest.approx(1 / 3)


def test_absent_ground_truth_classes_do_not_cap_macro_f1():
    report = []
    metrics = _label_ranking_metrics(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["A", "B"],
        ("A", "B", "C"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        per_class_out=report,
    )

    assert metrics["accuracy"] == metrics["f1_macro"] == 1.0
    assert metrics["prediction_coverage"] == pytest.approx(2 / 3)
    assert report[2] == {
        "class_id": 2,
        "label": "C",
        "support": 0,
        "predicted": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "recall_at_5": 0.0,
    }


def test_prototype_recall_at_five_uses_ranked_classes():
    metrics = _label_ranking_metrics(
        torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]),
        ["E"],
        ("A", "B", "C", "D", "E", "F"),
        torch.eye(6),
    )

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0


def test_macro_f1_exposes_single_class_prediction_collapse():
    labels = ("normal", "a", "b", "c", "d", "e", "f")
    true = ["normal"] * 5 + ["a", "b", "c", "d", "e", "f", "a"]
    z = torch.tensor([[1.0, 0.0]] * len(true))
    prototypes = torch.tensor([[1.0, 0.0]] + [[0.0, 1.0]] * (len(labels) - 1))

    metrics = _label_ranking_metrics(z, true, labels, prototypes)

    assert metrics["accuracy"] == pytest.approx(5 / 12)
    assert metrics["f1_macro"] == pytest.approx((10 / 17) / 7)
    assert metrics["prediction_coverage"] == pytest.approx(1 / 7)


def test_prediction_metrics_are_uniform_per_family_and_equal_family_overall():
    labels = ["a", "b"] * 3
    families = ["smellnet", "smellnet", "ecg", "ecg", "tactile", "tactile"]
    z = torch.eye(6)
    bank = {
        "smellnet": (("a", "b"), z[[0, 1]]),
        "ecg": (("a", "b"), z[[2, 3]]),
        # Reverse tactile candidates so its top-1 metrics are zero while
        # Recall@5 remains one for this two-candidate example.
        "tactile": (("a", "b"), z[[5, 4]]),
    }
    reports = {}
    metrics = _ts_prediction_metrics(z, labels, families, bank, reports)

    for family in ("smellnet", "ecg", "tactile"):
        for stat in ("accuracy", "f1_macro", "recall_at_1", "recall_at_5", "map"):
            assert f"{stat}/ts_{family}" in metrics
    assert metrics["accuracy/ts_tactile"] == 0.0
    assert metrics["f1_macro/ts_tactile"] == 0.0
    assert metrics["recall_at_1/ts_tactile"] == 0.0
    assert metrics["recall_at_5/ts_tactile"] == 1.0
    assert metrics["map/ts_tactile"] == 0.5
    assert metrics["accuracy/overall"] == pytest.approx(2 / 3)
    assert metrics["f1_macro/overall"] == pytest.approx(2 / 3)
    assert metrics["recall_at_1/overall"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5/overall"] == 1.0
    assert metrics["map/overall"] == pytest.approx(5 / 6)
    assert set(reports) == {"smellnet", "ecg"}


def test_validation_metrics_keep_a_compact_core():
    metrics = _metric_groups(
        "val",
        loss_metrics={
            "loss/siglip": 0.15,
            "loss/ts_smellnet": 0.14,
            "loss/ts_ecg": 0.16,
            "loss/ts_tactile": 0.15,
            "loss/distill": 0.1,
            "loss/total": 0.25,
        },
        counts={
            "n/img_image": 2,
            "n/img_video": 1,
            "n/ts_signal": 9,
            "n/ts_smellnet": 3,
            "n/ts_ecg": 3,
            "n/ts_tactile": 3,
        },
        prediction_metrics={
            "accuracy/ts_smellnet": 0.5,
            "f1_macro/ts_smellnet": 0.4,
            "recall_at_1/ts_smellnet": 0.5,
            "recall_at_5/ts_smellnet": 0.8,
            "prediction_coverage/ts_smellnet": 0.25,
            "accuracy/overall": 0.5,
            "f1_macro/overall": 0.4,
            "recall_at_1/overall": 0.3,
            "recall_at_5/overall": 0.7,
            "map/overall": 0.45,
            "accuracy/ts_tactile": 0.1,
            "f1_macro/ts_tactile": 0.05,
            "recall_at_1/ts_tactile": 0.1,
            "recall_at_5/ts_tactile": 0.3,
            "map/ts_tactile": 0.2,
            "prediction_coverage/ts_tactile": 0.4,
        },
    )

    assert metrics["val-core/accuracy/smellnet"] == 0.5
    assert metrics["val-core/f1_macro/smellnet"] == 0.4
    assert metrics["val-core/recall_at_5/smellnet"] == 0.8
    assert metrics["val-core/accuracy/overall"] == 0.5
    assert metrics["val-core/f1_macro/overall"] == 0.4
    assert metrics["val-core/recall_at_1/overall"] == 0.3
    assert metrics["val-core/recall_at_5/overall"] == 0.7
    assert metrics["val-core/map/overall"] == 0.45
    assert metrics["val-core/accuracy/tactile"] == 0.1
    assert metrics["val-core/f1_macro/tactile"] == 0.05
    assert metrics["val-core/recall_at_1/tactile"] == 0.1
    assert metrics["val-core/recall_at_5/tactile"] == 0.3
    assert metrics["val-core/map/tactile"] == 0.2
    assert metrics["val-aux/prediction_coverage/smellnet"] == 0.25
    assert metrics["val-aux/prediction_coverage/tactile"] == 0.4
    assert metrics["val-core/loss/aggregate"] == 0.25
    assert metrics["val-core/loss/smellnet"] == 0.14
    assert metrics["val-core/loss/ecg"] == 0.16
    assert metrics["val-core/loss/tactile"] == 0.15
    assert metrics["val-aux/loss/siglip"] == 0.15
    assert metrics["val-aux/loss/distill"] == 0.1


def test_training_metrics_publish_only_core_scores_and_actionable_diagnostics():
    metrics = {
        "accuracy/ts_smellnet": 0.5,
        "f1_macro/ts_smellnet": 0.4,
        "recall_at_1/ts_smellnet": 0.5,
        "recall_at_5/ts_smellnet": 0.8,
        "accuracy/overall": 0.5,
        "f1_macro/overall": 0.4,
        "recall_at_1/overall": 0.45,
        "recall_at_5/overall": 0.75,
        "map/overall": 0.55,
        "prediction_coverage/ts_smellnet": 0.2,
        "loss/siglip": 0.75,
        "loss/ts_smellnet": 0.7,
        "loss/ts_ecg": 0.8,
        "loss/ts_tactile": 0.75,
        "loss/distill": 0.5,
        "loss/total": 1.25,
        "recall_at_1/ts_tactile": 0.25,
        "recall_at_5/ts_tactile": 0.75,
        "map/ts_tactile": 0.5,
        "accuracy/ts_tactile": 0.25,
        "f1_macro/ts_tactile": 0.2,
        "prediction_coverage/ts_tactile": 0.1,
        "grad_norm": 2.0,
        "logit_scale": 10.0,
    }
    counts = Counter()
    counts.update(
        {
            "n/img_image": 2,
            "n/img_video": 1,
            "n/ts_smellnet": 8,
            "n/skipped_image": 1,
            "n/skipped_video": 2,
        }
    )

    grouped = _metric_groups("train", metrics, counts)

    assert grouped["train-core/accuracy/smellnet"] == 0.5
    assert grouped["train-core/f1_macro/overall"] == 0.4
    assert grouped["train-core/recall_at_1/overall"] == 0.45
    assert grouped["train-core/recall_at_5/overall"] == 0.75
    assert grouped["train-core/map/overall"] == 0.55
    assert grouped["train-core/recall_at_5/smellnet"] == 0.8
    assert grouped["train-core/accuracy/tactile"] == 0.25
    assert grouped["train-core/f1_macro/tactile"] == 0.2
    assert grouped["train-core/recall_at_1/tactile"] == 0.25
    assert grouped["train-core/recall_at_5/tactile"] == 0.75
    assert grouped["train-core/map/tactile"] == 0.5
    assert grouped["train-aux/prediction_coverage/smellnet"] == 0.2
    assert grouped["train-aux/prediction_coverage/tactile"] == 0.1
    assert grouped["train-aux/n/img"] == 3.0
    assert grouped["train-aux/n/skipped/image"] == 1.0
    assert grouped["train-aux/n/skipped/video"] == 2.0
    assert grouped["train-aux/n/skipped/signal"] == 0.0
    assert grouped["train-aux/n/skipped/total"] == 3.0
    assert grouped["train-aux/skipped_fraction"] == 3 / 14
    assert grouped["train-core/loss/aggregate"] == 1.25
    assert grouped["train-core/loss/smellnet"] == 0.7
    assert grouped["train-core/loss/ecg"] == 0.8
    assert grouped["train-core/loss/tactile"] == 0.75
    assert grouped["train-aux/loss/siglip"] == 0.75
    assert grouped["train-aux/loss/distill"] == 0.5


def test_loss_registration_and_batch_counts_fail_closed():
    with pytest.raises(RuntimeError, match="_REDUCED_METRIC_KEYS"):
        _allreduce_metrics({"loss/typo": 0.5}, torch.device("cpu"), world_size=1)

    counts = Counter()
    add_batch_counts(
        counts,
        {
            "kind": "signal",
            "media": [1, 2, 3],
            "family": "smellnet",
            "skipped": {"image": 0, "video": 0, "signal": 1},
        },
    )
    assert counts == {"n/ts_smellnet": 3, "n/skipped_signal": 1}
