# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mirl_ext.alignment.losses import (  # noqa: E402
    distill_cosine,
    siglip_sigmoid,
)
from mirl_ext.alignment.metrics import (  # noqa: E402
    _allreduce_metrics,
    _effective_dim,
    _label_ranking_metrics,
    _training_metric_groups,
    _ts_prediction_metrics,
    _validation_metric_groups,
    add_ts_family_counts,
    new_counts,
)
from mirl_ext.alignment.objective import _family_label_siglip_loss  # noqa: E402


def _scalars(positive_rate: float = 0.25):
    return torch.tensor(math.log(10.0)), torch.tensor(math.log(positive_rate / (1 - positive_rate)))


def test_siglip_supports_duplicate_positives_and_rectangular_prototypes():
    anchors = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    prototypes = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    targets = torch.tensor([0, 0, 1, 2, 2])
    pos_mask = targets[:, None] == torch.arange(3)[None, :]
    row_weights = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5])[:, None]
    scale, bias = _scalars(1 / 3)

    loss = siglip_sigmoid(
        anchors,
        prototypes,
        scale,
        bias,
        pos_mask=pos_mask,
        pair_weight=row_weights,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_siglip_mean_reduction_stays_order_one():
    scale, bias = _scalars()
    for batch in (8, 32, 64):
        torch.manual_seed(batch)
        a = torch.nn.functional.normalize(torch.randn(batch, 32), dim=-1)
        b = torch.nn.functional.normalize(torch.randn(batch, 32), dim=-1)
        labels = torch.arange(batch) % 4
        loss = siglip_sigmoid(a, b, scale, bias, labels[:, None] == labels[None, :])
        assert 0.1 < float(loss) < 3.0


def test_distill_cosine_uses_direction_and_detaches_teacher():
    student = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0]], requires_grad=True)
    assert distill_cosine(student, teacher).detach().item() == 0.0
    distill_cosine(-student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_token_distillation_weights_samples_not_token_count():
    student = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    # Sample one has one perfect token; sample two has two opposite tokens. Equal
    # sample weighting gives (0 + 2) / 2 = 1 rather than the row mean 4/3.
    assert float(distill_cosine(student, teacher, [1, 2])) == pytest.approx(1.0)


def test_family_label_loss_is_class_balanced():
    pa = torch.tensor([1.0, 0.0])
    pb = torch.tensor([0.0, 1.0])
    scale, _ = _scalars(0.5)

    def compute(n_a: int):
        z = torch.stack([pa] * n_a + [pb])
        return _family_label_siglip_loss(
            z,
            ["A"] * n_a + ["B"],
            ["ecg"] * (n_a + 1),
            {"ecg": (("A", "B"), torch.stack([pa, pb]))},
            scale,
        )

    small_families = compute(2)
    large_families = compute(20)
    assert float(small_families["ecg"]) == pytest.approx(
        float(large_families["ecg"]), rel=1e-6
    )


def test_family_label_loss_uses_absent_labels_as_negatives():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scale, _ = _scalars(1 / 3)
    family_losses = _family_label_siglip_loss(
        anchors,
        ["A", "B"],
        ["ecg", "ecg"],
        {"ecg": (("A", "B", "C"), prototypes)},
        scale,
    )
    assert torch.isfinite(family_losses["ecg"])
    assert family_losses.keys() == {"ecg"}


def test_family_label_bank_handles_unique_tactile_answers(monkeypatch):
    captured = {}

    def capture_siglip(a, b, _scale, _bias, pos_mask, pair_weight=None):
        captured.update(pos_mask=pos_mask, pair_weight=pair_weight)
        return a.sum() * 0.0 + b.sum() * 0.0

    monkeypatch.setattr("mirl_ext.alignment.objective.siglip_sigmoid", capture_siglip)

    captions = ["the cup slips during the lift", "the grasp remains stable"]
    losses = _family_label_siglip_loss(
        torch.eye(2),
        captions,
        ["tactile", "tactile"],
        {"tactile": (tuple(captions), torch.eye(2))},
        torch.tensor(math.log(10.0)),
    )

    assert losses.keys() == {"tactile"}
    assert torch.isfinite(losses["tactile"])
    assert captured["pos_mask"].tolist() == [[True, False], [False, True]]
    assert torch.equal(captured["pair_weight"], torch.ones(2, 1))


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


def test_prediction_metrics_are_per_family_and_overall_only_supervised_families():
    labels = ["a", "b"] * 3
    families = ["smellnet", "smellnet", "ecg", "ecg", "tactile", "tactile"]
    z = torch.eye(6)
    bank = {
        family: (("a", "b"), z[[start, start + 1]])
        for family, start in (("smellnet", 0), ("ecg", 2), ("tactile", 4))
    }
    reports = {}
    metrics = _ts_prediction_metrics(z, labels, families, bank, reports)

    for family in ("smellnet", "ecg"):
        assert metrics[f"accuracy/ts_{family}"] == 1.0
        assert metrics[f"f1_macro/ts_{family}"] == 1.0
    assert "accuracy/ts_tactile" not in metrics
    assert "f1_macro/ts_tactile" not in metrics
    assert metrics["recall_at_1/ts_tactile"] == 1.0
    assert metrics["recall_at_5/ts_tactile"] == 1.0
    assert metrics["map/ts_tactile"] == 1.0
    assert metrics["accuracy/overall"] == metrics["f1_macro/overall"] == 1.0
    assert set(reports) == {"smellnet", "ecg"}


def test_validation_metrics_keep_a_compact_core():
    metrics = _validation_metric_groups(
        averaged_metrics={
            "loss/siglip": 0.15,
            "loss/ts_smellnet": 0.14,
            "loss/ts_ecg": 0.16,
            "loss/ts_tactile": 0.15,
            "loss/distill": 0.1,
            "loss/total": 0.25,
        },
        prediction_metrics={
            "accuracy/ts_smellnet": 0.5,
            "f1_macro/ts_smellnet": 0.4,
            "recall_at_5/ts_smellnet": 0.8,
            "prediction_coverage/ts_smellnet": 0.25,
            "accuracy/overall": 0.5,
            "f1_macro/overall": 0.4,
            "eff_dim/ts_tactile": 3.0,
            "recall_at_1/ts_tactile": 0.1,
            "recall_at_5/ts_tactile": 0.3,
            "map/ts_tactile": 0.2,
        },
        bucket_totals={
            "n/img_image": 2,
            "n/img_video": 1,
            "n/ts_signal": 9,
            "n/ts_smellnet": 3,
            "n/ts_ecg": 3,
            "n/ts_tactile": 3,
        },
    )

    assert metrics["val-core/accuracy/smellnet"] == 0.5
    assert metrics["val-core/f1_macro/smellnet"] == 0.4
    assert metrics["val-core/recall_at_5/smellnet"] == 0.8
    assert metrics["val-core/accuracy/overall"] == 0.5
    assert metrics["val-core/f1_macro/overall"] == 0.4
    assert "val-core/f1_macro/tactile" not in metrics
    assert metrics["val-core/recall_at_1/tactile"] == 0.1
    assert metrics["val-core/recall_at_5/tactile"] == 0.3
    assert metrics["val-core/map/tactile"] == 0.2
    assert metrics["val-aux/prediction_coverage/smellnet"] == 0.25
    assert metrics["val-aux/effective_dimension/tactile"] == 3.0
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
        "recall_at_5/ts_smellnet": 0.8,
        "accuracy/overall": 0.5,
        "f1_macro/overall": 0.4,
        "eff_dim/ts_smellnet": 3.0,
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
        "grad_norm": 2.0,
        "logit_scale": 10.0,
    }
    counts = new_counts()
    counts.update({"n/img_image": 2, "n/img_video": 1, "n/ts_smellnet": 8})

    grouped = _training_metric_groups(metrics, counts)

    assert grouped["train-core/accuracy/smellnet"] == 0.5
    assert grouped["train-core/f1_macro/overall"] == 0.4
    assert grouped["train-core/recall_at_5/smellnet"] == 0.8
    assert grouped["train-core/recall_at_1/tactile"] == 0.25
    assert grouped["train-core/recall_at_5/tactile"] == 0.75
    assert grouped["train-core/map/tactile"] == 0.5
    assert grouped["train-aux/effective_dimension/smellnet"] == 3.0
    assert grouped["train-aux/prediction_coverage/smellnet"] == 0.2
    assert grouped["train-aux/n/img"] == 3.0
    assert grouped["train-core/loss/aggregate"] == 1.25
    assert grouped["train-core/loss/smellnet"] == 0.7
    assert grouped["train-core/loss/ecg"] == 0.8
    assert grouped["train-core/loss/tactile"] == 0.75
    assert grouped["train-aux/loss/siglip"] == 0.75
    assert grouped["train-aux/loss/distill"] == 0.5


def test_effective_dim_detects_rank_collapse():
    direction = torch.nn.functional.normalize(torch.randn(16), dim=-1)
    collapsed = torch.arange(1.0, 9.0).unsqueeze(-1) * direction
    assert _effective_dim(collapsed) == pytest.approx(1.0, abs=0.05)

    torch.manual_seed(0)
    assert _effective_dim(torch.randn(256, 16)) > 12.0
    assert _effective_dim(torch.zeros(8, 4)) is None


def test_loss_registration_and_family_counts_fail_closed():
    with pytest.raises(RuntimeError, match="_REDUCED_METRIC_KEYS"):
        _allreduce_metrics({"loss/typo": 0.5}, torch.device("cpu"), world_size=1)

    counts = new_counts()
    add_ts_family_counts(counts, {"ts_format": ["smellnet", "smellnet", "ecg", "tactile"]})
    assert counts["n/ts_smellnet"] == 2
    assert counts["n/ts_ecg"] == counts["n/ts_tactile"] == 1
