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
    _prototype_classification_metrics,
    _training_metric_groups,
    _ts_prediction_metrics,
    _validation_metric_groups,
    add_ts_family_counts,
    new_counts,
)
from mirl_ext.alignment.objective import _family_prototype_siglip_loss  # noqa: E402
from mirl_ext.alignment.projection import GCMSMLPEncoder, GCMSProjectionHead, ProjectionHead  # noqa: E402


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


def test_projection_head_is_linear_unless_hidden_dimension_is_requested():
    linear = ProjectionHead(16, 8, hidden_dim=None)
    mlp = ProjectionHead(16, 8, hidden_dim=12)

    assert isinstance(linear.net, torch.nn.Linear)
    assert isinstance(mlp.net, torch.nn.Sequential)
    assert linear(torch.randn(3, 16)).shape == mlp(torch.randn(3, 16)).shape == (3, 8)



def test_gcms_projection_uses_smellnet_encoder_before_shared_adapter():
    gcms = GCMSProjectionHead(460, 32)
    layers = list(gcms.encoder.net)

    assert isinstance(gcms.encoder, GCMSMLPEncoder)
    assert [type(layer) for layer in layers] == [
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Dropout,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Dropout,
        torch.nn.Linear,
    ]
    assert [layer.out_features for layer in layers if isinstance(layer, torch.nn.Linear)] == [512, 256, 256]
    assert gcms.adapter.in_features == 256
    assert gcms(torch.randn(4, 460)).shape == (4, 32)


def test_family_prototype_loss_is_class_balanced():
    pa = torch.tensor([1.0, 0.0])
    pb = torch.tensor([0.0, 1.0])
    scale, _ = _scalars(0.5)

    def compute(n_a: int):
        z = torch.stack([pa] * n_a + [pb])
        return _family_prototype_siglip_loss(
            z,
            ["A"] * n_a + ["B"],
            ["ecg"] * (n_a + 1),
            {"ecg": (("A", "B"), torch.stack([pa, pb]))},
            ("ecg",),
            scale,
        )

    small, small_rate, small_coverage = compute(2)
    large, large_rate, large_coverage = compute(20)
    assert float(small) == pytest.approx(float(large), rel=1e-6)
    assert small_rate == large_rate == pytest.approx(0.5)
    assert small_coverage == large_coverage == {"ecg": 1.0}


def test_family_prototype_loss_uses_absent_classes_as_negatives():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scale, _ = _scalars(1 / 3)
    loss, rate, coverage = _family_prototype_siglip_loss(
        anchors,
        ["A", "B"],
        ["ecg", "ecg"],
        {"ecg": (("A", "B", "C"), prototypes)},
        ("ecg",),
        scale,
    )
    assert torch.isfinite(loss)
    assert rate == pytest.approx(1 / 3)
    assert coverage == {"ecg": 1.0}


def test_prototype_metrics_exclude_unknown_labels_and_report_coverage():
    metrics = _prototype_classification_metrics(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["A", "unseen"],
        ("A", "B"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert metrics["label_coverage"] == pytest.approx(0.5)
    assert metrics["class_coverage"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_effective_batch_macro_f1_is_not_an_average_of_micro_batches():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    first = _prototype_classification_metrics(
        torch.tensor([[1.0, 0.0]]),
        ["A"],
        ("A", "B"),
        prototypes,
    )
    second = _prototype_classification_metrics(
        torch.tensor([[1.0, 0.0]]),
        ["B"],
        ("A", "B"),
        prototypes,
    )
    combined = _prototype_classification_metrics(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ["A", "B"],
        ("A", "B"),
        prototypes,
    )

    assert (first["f1_macro"] + second["f1_macro"]) / 2 == pytest.approx(0.5)
    assert combined["f1_macro"] == pytest.approx(1 / 3)


def test_absent_ground_truth_classes_do_not_cap_macro_f1():
    report = []
    metrics = _prototype_classification_metrics(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["A", "B"],
        ("A", "B", "C"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        per_class_out=report,
    )

    assert metrics["accuracy"] == metrics["f1_macro"] == 1.0
    assert metrics["class_coverage"] == metrics["prediction_coverage"] == pytest.approx(2 / 3)
    assert report[2] == {
        "class_id": 2,
        "label": "C",
        "support": 0,
        "predicted": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_macro_f1_exposes_single_class_prediction_collapse():
    labels = ("normal", "a", "b", "c", "d", "e", "f")
    true = ["normal"] * 5 + ["a", "b", "c", "d", "e", "f", "a"]
    z = torch.tensor([[1.0, 0.0]] * len(true))
    prototypes = torch.tensor([[1.0, 0.0]] + [[0.0, 1.0]] * (len(labels) - 1))

    metrics = _prototype_classification_metrics(z, true, labels, prototypes)

    assert metrics["accuracy"] == pytest.approx(5 / 12)
    assert metrics["f1_macro"] == pytest.approx((10 / 17) / 7)
    assert metrics["prediction_coverage"] == pytest.approx(1 / 7)


def test_prediction_metrics_are_per_family_and_overall_only_supervised_families():
    labels = ["a", "b"] * 3
    families = ["smell", "smell", "ecg", "ecg", "tactile", "tactile"]
    z = torch.eye(6)
    bank = {family: (("a", "b"), z[[start, start + 1]]) for family, start in (("smell", 0), ("ecg", 2))}
    reports = {}
    metrics = _ts_prediction_metrics(z, labels, families, bank, reports)

    for family in ("smell", "ecg"):
        assert metrics[f"accuracy/ts_{family}"] == 1.0
        assert metrics[f"f1_macro/ts_{family}"] == 1.0
    assert "accuracy/ts_tactile" not in metrics
    assert "f1_macro/ts_tactile" not in metrics
    assert metrics["f1_macro/ts_supervised_family_macro"] == 1.0
    assert set(reports) == {"smell", "ecg"}


def test_validation_metrics_separate_core_from_aux_and_do_not_fake_haptic_f1():
    metrics = _validation_metric_groups(
        averaged_metrics={"loss/total": 0.25},
        prediction_metrics={
            "accuracy/ts_smell": 0.5,
            "accuracy/smell_sensor_to_gcms": 0.6,
            "accuracy/smell_gcms_to_text": 0.8,
            "f1_macro/ts_smell": 0.4,
            "class_coverage/ts_smell": 0.75,
            "prediction_coverage/ts_smell": 0.25,
            "accuracy/ts_supervised_family_macro": 0.5,
            "f1_macro/ts_supervised_family_macro": 0.4,
            "eff_dim/ts_tactile": 3.0,
        },
        bucket_totals={
            "n/img_image": 2,
            "n/img_video": 1,
            "n/ts_signal": 9,
            "n/ts_smell": 3,
            "n/ts_ecg": 3,
            "n/ts_tactile": 3,
        },
    )

    assert metrics["val-core/accuracy/smellnet"] == 0.5
    assert metrics["val-core/f1_macro/smellnet"] == 0.4
    assert metrics["val-aux/accuracy/smellnet_sensor_to_gcms"] == 0.6
    assert metrics["val-aux/accuracy/smellnet_gcms_to_text"] == 0.8
    assert metrics["val-core/accuracy/overall"] == 0.5
    assert metrics["val-core/f1_macro/overall"] == 0.4
    assert "val-core/f1_macro/haptic" not in metrics
    assert metrics["val-aux/prediction_coverage/smellnet"] == 0.25
    assert metrics["val-aux/effective_dimension/haptic"] == 3.0
    assert metrics["val/loss"] == 0.25


def test_training_metrics_publish_only_core_scores_and_actionable_diagnostics():
    metrics = {
        "accuracy/ts_smell": 0.5,
        "f1_macro/ts_smell": 0.4,
        "accuracy/ts_supervised_family_macro": 0.5,
        "f1_macro/ts_supervised_family_macro": 0.4,
        "eff_dim/ts_smell": 3.0,
        "prediction_coverage/ts_smell": 0.2,
        "gap/ts_smell": 0.1,
        "loss/total": 1.25,
        "grad_norm/vit": 2.0,
        "logit_scale": 10.0,
    }
    counts = new_counts()
    counts.update({"n/img_image": 2, "n/img_video": 1, "n/ts_smell": 8})

    grouped = _training_metric_groups(metrics, counts)

    assert grouped["train-core/accuracy/smellnet"] == 0.5
    assert grouped["train-core/f1_macro/overall"] == 0.4
    assert grouped["train-aux/effective_dimension/smellnet"] == 3.0
    assert grouped["train-aux/prediction_coverage/smellnet"] == 0.2
    assert grouped["train-aux/n/img"] == 3.0
    assert grouped["train/loss"] == 1.25
    assert not any("gap" in key for key in grouped)


def test_effective_dim_detects_rank_collapse():
    direction = torch.nn.functional.normalize(torch.randn(16), dim=-1)
    collapsed = torch.arange(1.0, 9.0).unsqueeze(-1) * direction
    assert _effective_dim(collapsed) == pytest.approx(1.0, abs=0.05)

    torch.manual_seed(0)
    assert _effective_dim(torch.randn(256, 16)) > 12.0
    assert _effective_dim(torch.zeros(8, 4)) is None


def test_metric_registration_and_family_counts_fail_closed():
    with pytest.raises(RuntimeError, match="_REDUCED_METRIC_KEYS"):
        _allreduce_metrics({"accuracy/ts_typo": 0.5}, torch.device("cpu"), world_size=1)

    counts = new_counts()
    add_ts_family_counts(counts, {"ts_format": ["smell", "smell", "ecg", "tactile"]})
    assert counts["n/ts_smell"] == 2
    assert counts["n/ts_ecg"] == counts["n/ts_tactile"] == 1
