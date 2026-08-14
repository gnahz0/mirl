
from __future__ import annotations

import math
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from mirl_ext.alignment.data import (  # noqa: E402
    TACTILE_NUM_LABELS,
    TACTILE_SPANS,
    TASK_LABELS,
)
from mirl_ext.alignment.metrics import (  # noqa: E402
    _bank_metrics,
    _bank_stats,
    _merge_prediction_metrics,
    _metric_groups,
    build_bank_specs,
    new_stats,
    prediction_metrics,
    update_stats,
)
from mirl_ext.alignment.objective import (  # noqa: E402
    _build_tactile_bank,
    _compute_losses,
    _label_siglip_loss,
    _tactile_siglip_loss,
)


def _signal_batch(size: int) -> dict:
    return {
        "kind": "signal",
        "media": [0] * size,
        "family": "smellnet",
        "skipped": {"image": 0, "video": 0, "signal": 1},
    }


def _rank(z, labels, candidates, bank, rows_out=None):
    """Score one exclusive family the way the trainer does: one-hot into one bank."""
    spec = build_bank_specs({"ecg": (candidates, bank)}, None)[0]
    ids = torch.tensor([candidates.index(label) for label in labels])
    target = torch.nn.functional.one_hot(ids, num_classes=len(candidates))
    return _bank_metrics(_bank_stats(z, target, spec), spec, rows_out=rows_out)


def _score_units(specs, ts_eval, log_logit_scale=None, logit_bias=None, **reports):
    """Stream one microbatch through the unified path and derive every metric."""
    stats = new_stats(specs, torch.device("cpu"))
    update_stats(stats, specs, ts_eval, log_logit_scale, logit_bias)
    return prediction_metrics(stats, specs, **reports)


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
            torch.tensor(-1.0),
        )

    assert float(compute(2)) == pytest.approx(float(compute(20)), rel=1e-6)


def test_label_loss_sums_candidates_before_averaging_anchors():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    candidate_labels = ("A", "B", "C")
    labels = ["A", "B"]
    log_scale = torch.tensor(math.log(2.0))
    bias = torch.tensor(-0.75)

    loss = _label_siglip_loss(
        anchors,
        labels,
        candidate_labels,
        prototypes,
        log_scale,
        bias,
    )

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
        torch.tensor(-10.0),
    )
    assert torch.isfinite(loss)


def test_siglip_scale_and_bias_are_both_trainable():
    scale = torch.nn.Parameter(torch.tensor(math.log(10.0)))
    bias = torch.nn.Parameter(torch.tensor(-10.0))
    loss = _label_siglip_loss(
        torch.eye(2),
        ["A", "B"],
        ("A", "B"),
        torch.eye(2),
        scale,
        bias,
    )

    loss.backward()

    assert scale.grad is not None and scale.grad != 0
    assert bias.grad is not None and bias.grad != 0


def test_structured_tactile_loss_accepts_multiple_positive_labels():
    anchors = torch.tensor([[2**-0.5, 2**-0.5]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    loss, per_task = _tactile_siglip_loss(
        anchors,
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.ones(1, 3),
        prototypes,
        log_logit_scale=torch.tensor(math.log(10.0)),
        logit_bias=torch.tensor(-1.0),
    )

    assert loss is not None and torch.isfinite(loss)
    assert per_task == {} or all(math.isfinite(v) for v in per_task.values())


def _reference_per_task_mean(z, targets, masks, embeddings, scale, bias):
    """The pre-refactor objective: one masked BCE per task, then a plain mean.
    Written the way the six-call version was, so it is an independent oracle
    rather than a restatement of the implementation."""
    losses = []
    for start, stop in TACTILE_SPANS.values():
        rows = masks[:, start] > 0
        if not bool(rows.any()):
            continue
        logits = scale.exp() * (z[rows] @ embeddings[start:stop].t()) + bias
        total = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets[rows, start:stop], reduction="sum")
        losses.append(total / (int(rows.sum()) * (stop - start)))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("partial", [False, True])
def test_unified_tactile_loss_matches_per_task_reference(partial):
    """The refactor is plumbing only: same number, across all six mixed-width tasks."""
    torch.manual_seed(0)
    width = TACTILE_NUM_LABELS
    z = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    embeddings = torch.nn.functional.normalize(torch.randn(width, 8), dim=-1)
    targets = (torch.rand(5, width) > 0.5).float()
    masks = torch.ones(5, width)
    if partial:
        # Row 3 answers only the first task; row 4 answers nothing -- the two real shapes of incomplete join.
        first_stop = next(iter(TACTILE_SPANS.values()))[1]
        masks[3, first_stop:] = 0.0
        masks[4] = 0.0
    scale = torch.tensor(math.log(10.0))
    bias = torch.tensor(-0.5)

    actual, per_task = _tactile_siglip_loss(z, targets, masks, embeddings, scale, bias)
    expected = _reference_per_task_mean(z, targets, masks, embeddings, scale, bias)

    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
    assert len(per_task) == len(TACTILE_SPANS)


def test_unmasked_labels_contribute_no_gradient():
    """A row with no annotation for a task must not push that task's labels: its
    all-zero target would otherwise read under BCE as "every label is absent"."""
    torch.manual_seed(0)
    width = TACTILE_NUM_LABELS
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1).requires_grad_(True)
    embeddings = torch.nn.functional.normalize(torch.randn(width, 8), dim=-1)
    targets = torch.zeros(3, width)
    masks = torch.ones(3, width)
    masks[2] = 0.0  # row 2 has no annotation for any task

    loss, per_task = _tactile_siglip_loss(
        z, targets, masks, embeddings, torch.tensor(math.log(10.0)), torch.tensor(-1.0)
    )
    loss.backward()

    assert torch.equal(z.grad[2], torch.zeros(8)), "a fully masked row must not move"
    assert (z.grad[:2] != 0).any(), "annotated rows must still receive gradient"
    # The divisor counts known pairs only, so masking must not silently rescale surviving rows.
    assert loss.item() == pytest.approx(
        _tactile_siglip_loss(
            z.detach()[:2],
            targets[:2],
            masks[:2],
            embeddings,
            torch.tensor(math.log(10.0)),
            torch.tensor(-1.0),
        )[0].item(),
        rel=1e-6,
    )
    assert set(per_task) == {f"loss/task/{task}" for task in TACTILE_SPANS}


def test_general_objective_averages_structured_tasks_only_for_tactile():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_logit_scale = torch.nn.Parameter(torch.tensor(math.log(2.0)))
            self.logit_bias = torch.nn.Parameter(torch.tensor(-1.0))

        def forward(self, kind, media, family, max_image_tokens):
            assert (kind, family, max_image_tokens) == ("signal", "tactile", 1024)
            return None, None, None, torch.eye(2), self.log_logit_scale, self.logit_bias

    width = TACTILE_NUM_LABELS
    bank = (
        tuple(f"label{index}" for index in range(width)),
        torch.nn.functional.normalize(torch.arange(width * 2, dtype=torch.float32).reshape(width, 2), dim=-1),
    )
    batch = {
        "kind": "signal",
        "media": [None, None],
        "family": "tactile",
        "text": ["unused", "unused"],
        "targets": torch.zeros(2, width).index_fill_(1, torch.tensor([0]), 1.0),
        "masks": torch.ones(2, width),
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
        tactile_bank=bank,
    )

    # One loss for the whole family, and a per-task breakdown for every task.
    assert total.detach().item() == pytest.approx(metrics["loss/ts_tactile"])
    assert metrics["loss/siglip"] == pytest.approx(metrics["loss/ts_tactile"])
    assert {key for key in metrics if key.startswith("loss/task/")} == {f"loss/task/{task}" for task in TACTILE_SPANS}
    # ts_eval now carries the (B, 30) target/mask pair rather than per-task dicts.
    assert task_eval[3].shape == (2, TACTILE_NUM_LABELS)
    assert task_eval[4].shape == (2, TACTILE_NUM_LABELS)


def test_structured_tactile_metrics_merge_as_one_family():
    scale = torch.tensor(math.log(10.0))
    multi_z = torch.tensor([[2**-0.5, 2**-0.5]])
    multi_text = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-(2**-0.5), -(2**-0.5)]])
    # Only the first task is exercised; a wholly-unobserved task must be skipped rather than scored.
    width = TACTILE_NUM_LABELS
    first_stop = next(iter(TACTILE_SPANS.values()))[1]
    embeddings = torch.zeros(width, 2)
    embeddings[:3] = multi_text
    targets = torch.zeros(1, width)
    targets[0, :2] = 1.0
    masks = torch.zeros(1, width)
    masks[0, :first_stop] = 1.0
    specs = build_bank_specs(
        {},
        (tuple(f"label{index}" for index in range(width)), embeddings),
    )
    tactile = _score_units(
        specs,
        (multi_z, [], [], targets, masks),
        scale,
        torch.tensor(-1.0),
    )
    merged = _merge_prediction_metrics(
        {"map/ts_smellnet": 0.5, "map/overall": 0.5},
        tactile,
    )

    assert tactile["accuracy/task/initial_fingers"] == 1.0
    assert tactile["recall_at_1/ts_tactile"] == 0.5
    assert tactile["map/ts_tactile"] == 1.0
    assert merged["map/overall"] == pytest.approx(0.75)


def test_structured_tactile_bank_preserves_task_label_order():
    class TextModel:
        def encode_text(self, labels, device):
            return torch.eye(len(labels))

    labels, embeddings = _build_tactile_bank(TextModel(), torch.device("cpu"))

    assert len(labels) == TACTILE_NUM_LABELS
    assert embeddings.shape[0] == TACTILE_NUM_LABELS
    start, stop = TACTILE_SPANS["force_level"]
    assert labels[start:stop] == TASK_LABELS["force_level"]


def test_absent_ground_truth_classes_remain_in_report():
    report = []
    metrics = _rank(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["A", "B"],
        ("A", "B", "C"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        rows_out=report,
    )

    assert metrics["accuracy"] == 1.0
    assert report[2] == {
        "class_id": 2,
        "label": "C",
        "support": 0,
        "predicted": 0,
        "precision": 0.0,
        "recall": 0.0,
        "recall_at_5": 0.0,
    }


def test_prototype_recall_at_five_uses_ranked_classes():
    metrics = _rank(
        torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]),
        ["E"],
        ("A", "B", "C", "D", "E", "F"),
        torch.eye(6),
    )

    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0


def test_prediction_coverage_exposes_top1_collapse():
    labels = ("normal", "a", "b", "c", "d", "e", "f")
    metrics = _rank(
        torch.tensor([[1.0, 0.0]] * 12),
        ["normal"] * 5 + ["a", "b", "c", "d", "e", "f", "a"],
        labels,
        torch.tensor([[1.0, 0.0]] + [[0.0, 1.0]] * (len(labels) - 1)),
    )

    assert metrics["prediction_coverage"] == pytest.approx(1 / 7)


def test_prediction_metrics_are_uniform_per_family_and_equal_family_overall():
    labels = ["a", "b"] * 3
    families = ["smellnet", "smellnet", "ecg", "ecg", "tactile", "tactile"]
    z = torch.eye(6)
    bank = {
        "smellnet": (("a", "b"), z[[0, 1]]),
        "ecg": (("a", "b"), z[[2, 3]]),
        # Reverse tactile candidates so top-1 is zero while Recall@5 stays one.
        "tactile": (("a", "b"), z[[5, 4]]),
    }
    reports = {}
    specs = build_bank_specs(bank, None)
    metrics = _score_units(specs, (z, labels, families, None, None), per_class=reports)

    for family in ("smellnet", "ecg", "tactile"):
        for stat in ("accuracy", "recall_at_1", "recall_at_5", "map"):
            assert f"{stat}/ts_{family}" in metrics
        assert metrics[f"prediction_coverage/ts_{family}"] == 1.0
    assert metrics["accuracy/ts_tactile"] == 0.0
    assert metrics["recall_at_1/ts_tactile"] == 0.0
    assert metrics["recall_at_5/ts_tactile"] == 1.0
    assert metrics["map/ts_tactile"] == 0.5
    assert metrics["accuracy/overall"] == pytest.approx(2 / 3)
    assert metrics["recall_at_1/overall"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5/overall"] == 1.0
    assert metrics["map/overall"] == pytest.approx(5 / 6)
    assert metrics["prediction_coverage/overall"] == 1.0
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
            "recall_at_1/ts_smellnet": 0.5,
            "recall_at_5/ts_smellnet": 0.8,
            "prediction_coverage/ts_smellnet": 0.2,
            "accuracy/overall": 0.5,
            "recall_at_1/overall": 0.3,
            "recall_at_5/overall": 0.7,
            "map/overall": 0.45,
            "prediction_coverage/overall": 0.3,
            "accuracy/ts_tactile": 0.1,
            "recall_at_1/ts_tactile": 0.1,
            "recall_at_5/ts_tactile": 0.3,
            "map/ts_tactile": 0.2,
            "prediction_coverage/ts_tactile": 0.4,
        },
    )

    assert metrics["val-core/accuracy/smellnet"] == 0.5
    assert metrics["val-core/recall_at_5/smellnet"] == 0.8
    assert metrics["val-core/accuracy/overall"] == 0.5
    assert metrics["val-core/recall_at_1/overall"] == 0.3
    assert metrics["val-core/recall_at_5/overall"] == 0.7
    assert metrics["val-core/map/overall"] == 0.45
    assert metrics["val-core/accuracy/tactile"] == 0.1
    assert metrics["val-core/recall_at_1/tactile"] == 0.1
    assert metrics["val-core/recall_at_5/tactile"] == 0.3
    assert metrics["val-core/map/tactile"] == 0.2
    assert metrics["val-aux/prediction_coverage/smellnet"] == 0.2
    assert metrics["val-aux/prediction_coverage/tactile"] == 0.4
    assert metrics["val-aux/prediction_coverage/overall"] == 0.3
    assert metrics["val-core/loss/aggregate"] == 0.25
    assert metrics["val-core/loss/smellnet"] == 0.14
    assert metrics["val-core/loss/ecg"] == 0.16
    assert metrics["val-core/loss/tactile"] == 0.15
    assert metrics["val-aux/loss/siglip"] == 0.15
    assert metrics["val-aux/loss/distill"] == 0.1


def test_training_metrics_publish_only_core_scores_and_actionable_diagnostics():
    metrics = {
        "accuracy/ts_smellnet": 0.5,
        "recall_at_1/ts_smellnet": 0.5,
        "recall_at_5/ts_smellnet": 0.8,
        "prediction_coverage/ts_smellnet": 0.2,
        "accuracy/overall": 0.5,
        "recall_at_1/overall": 0.45,
        "recall_at_5/overall": 0.75,
        "map/overall": 0.55,
        "prediction_coverage/overall": 0.3,
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
        "prediction_coverage/ts_tactile": 0.4,
        "grad_norm": 2.0,
        "logit_scale": 10.0,
        "logit_bias": -10.0,
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

    grouped = _metric_groups("train", metrics, counts, metrics)

    assert grouped["train-core/accuracy/smellnet"] == 0.5
    assert grouped["train-core/recall_at_1/overall"] == 0.45
    assert grouped["train-core/recall_at_5/overall"] == 0.75
    assert grouped["train-core/map/overall"] == 0.55
    assert grouped["train-core/recall_at_5/smellnet"] == 0.8
    assert grouped["train-core/accuracy/tactile"] == 0.25
    assert grouped["train-core/recall_at_1/tactile"] == 0.25
    assert grouped["train-core/recall_at_5/tactile"] == 0.75
    assert grouped["train-core/map/tactile"] == 0.5
    assert grouped["train-aux/prediction_coverage/smellnet"] == 0.2
    assert grouped["train-aux/prediction_coverage/tactile"] == 0.4
    assert grouped["train-aux/prediction_coverage/overall"] == 0.3
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
    assert grouped["train-aux/logit_scale"] == 10.0
    assert grouped["train-aux/logit_bias"] == -10.0


def test_loss_registration_and_batch_counts_fail_closed():
    from mirl_ext.alignment.trainer import AccumulationWindow

    window = AccumulationWindow(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="_REDUCED_METRIC_KEYS"):
        window.add(_signal_batch(0), {"loss/typo": 0.5}, (None, [], [], None, None))

    window = AccumulationWindow(torch.device("cpu"))
    window.add(_signal_batch(3), {}, (None, [], [], None, None))
    _, counts, _ = window.flush()
    assert counts["n/ts_smellnet"] == 3
    assert counts["n/skipped_signal"] == 1
    assert counts["n/ts_signal"] == 3  # derived from the per-family totals


def test_window_averages_only_over_microbatches_that_produced_a_loss():
    from mirl_ext.alignment.trainer import AccumulationWindow

    window = AccumulationWindow(torch.device("cpu"))
    for index in range(4):
        payload = {"loss/total": 1.0}
        if index < 2:  # only half the window held a tactile batch
            payload["loss/ts_tactile"] = 3.0
        window.add(_signal_batch(1), payload, (None, [], [], None, None))
    losses, _, _ = window.flush()

    assert losses["loss/total"] == pytest.approx(1.0)
    assert losses["loss/ts_tactile"] == pytest.approx(3.0)
    # Losses nothing produced are dropped rather than logged as nan.
    assert "loss/ts_ecg" not in losses
