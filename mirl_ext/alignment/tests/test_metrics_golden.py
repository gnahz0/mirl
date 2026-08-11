# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Guard the metric surface established by the 2026-08-09 rewrite.

``fixtures/metrics_golden.json`` was produced by ``regen_metrics_golden.py`` run
against the implementation that preceded the rewrite -- the one that hand-packed
every statistic into a positional buffer and reduced it with torchmetrics
``MeanMetric``s alongside. It stores both the inputs and the outputs, so these
tests replay the same inputs through the new unified path and compare exactly.

Exact equality remains the assertion except for tactile multi-label F1 and
coverage, which now use canonical SigLIP's learned scalar bias instead of fixed
per-task priors. Ranking metrics remain unchanged by that calibration change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mirl_ext.alignment.data import MULTILABEL_TASKS, TACTILE_SPANS  # noqa: E402
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

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing golden fixture {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _single_label_spec(candidates, bank):
    return build_bank_specs({"ecg": (tuple(candidates), _tensor(bank))}, None)[0]


def _one_hot(labels, candidates) -> torch.Tensor:
    ids = torch.tensor([list(candidates).index(label) for label in labels])
    return torch.nn.functional.one_hot(ids, num_classes=len(candidates))


def _tactile_specs(case) -> tuple:
    return build_bank_specs({}, (tuple(case["labels"]), _tensor(case["bank"])))


def _assert_same(actual: dict, expected: dict, label: str) -> None:
    assert set(actual) == set(expected), f"{label}: key set changed"
    for key, value in expected.items():
        assert actual[key] == value, f"{label}: {key} moved {value!r} -> {actual[key]!r}"


def _assert_rows(actual: list[dict], expected: list[dict], label: str) -> None:
    assert len(actual) == len(expected), f"{label}: row count changed"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert got == want, f"{label}: row {index} changed"


@pytest.mark.parametrize("case", ["smell", "ecg"])
def test_single_label_families_score_unchanged(golden, case):
    """smellnet and ecg keep every scalar and every per-class row."""
    data = golden["inputs"][case]
    rows: list[dict] = []
    spec = _single_label_spec(data["candidates"], data["bank"])
    metrics = _bank_metrics(
        _bank_stats(_tensor(data["z"]), _one_hot(data["labels"], data["candidates"]), spec),
        spec,
        rows_out=rows,
    )
    name = "smellnet" if case == "smell" else "ecg"
    _assert_same(metrics, golden["golden"][name], name)
    _assert_rows(rows, golden["golden"][f"{name}_rows"], name)


def test_tactile_ranking_is_unchanged_and_classification_uses_learned_bias(golden):
    """Changing calibration leaves ranking intact and changes multi-label F1."""
    case = golden["inputs"]["tactile"]
    rows: list[dict] = []
    specs = _tactile_specs(case)
    stats = new_stats(specs, torch.device("cpu"))
    update_stats(
        stats,
        specs,
        (_tensor(case["z"]), [], [], _tensor(case["targets"]), _tensor(case["masks"])),
        torch.tensor(case["log_logit_scale"]),
        torch.tensor(-10.0),
    )
    metrics = prediction_metrics(stats, specs, per_label=rows)

    # The rewrite folded the equal-family rollup into prediction_metrics, so it now
    # emits the /overall keys the old caller derived in a second step. With tactile
    # the only family present, each one is just its ts_tactile value.
    expected = dict(golden["golden"]["tactile"])
    expected.update(
        {
            f"{stat}/overall": expected[f"{stat}/ts_tactile"]
            for stat in ("accuracy", "f1_macro", "recall_at_1", "recall_at_5", "map")
        }
    )
    ranking = {
        key: value
        for key, value in metrics.items()
        if not key.startswith("f1_macro/")
    }
    expected_ranking = {
        key: value
        for key, value in expected.items()
        if not key.startswith("f1_macro/")
    }
    _assert_same(ranking, expected_ranking, "tactile ranking")

    expected_rows = golden["golden"]["tactile_rows"]
    for actual, old in zip(rows, expected_rows, strict=True):
        for field in ("task", "class_id", "label", "support", "recall_at_5"):
            assert actual[field] == old[field]
        if actual["task"] in MULTILABEL_TASKS:
            assert actual["predicted"] == actual["precision"] == actual["recall"] == actual["f1"] == 0
        else:
            assert actual == old
    # local_shape was answered by no row, so it must be absent rather than nan.
    assert not any(key.endswith("/task/local_shape") for key in metrics)
    assert {row["task"] for row in rows} == set(TACTILE_SPANS) - {"local_shape"}


def test_mixed_families_and_overall_rollup_unchanged(golden):
    """Both single-label families in one pass, then the equal-family overall."""
    smell, ecg = golden["inputs"]["smell"], golden["inputs"]["ecg"]
    specs = build_bank_specs(
        {
            "smellnet": (tuple(smell["candidates"]), _tensor(smell["bank"])),
            "ecg": (tuple(ecg["candidates"]), _tensor(ecg["bank"])),
        },
        None,
    )
    reports: dict[str, list[dict]] = {}
    stats = new_stats(specs, torch.device("cpu"))
    update_stats(
        stats,
        specs,
        (
            torch.cat([_tensor(smell["z"]), _tensor(ecg["z"])]),
            [*smell["labels"], *ecg["labels"]],
            ["smellnet"] * len(smell["labels"]) + ["ecg"] * len(ecg["labels"]),
            None,
            None,
        ),
        None,
        None,
    )
    mixed = _merge_prediction_metrics(
        prediction_metrics(stats, specs, per_class=reports), golden["golden"]["tactile"]
    )

    _assert_same(mixed, golden["golden"]["mixed"], "mixed")
    assert sorted(reports) == golden["golden"]["mixed_report_families"]


def test_metric_groups_surface_unchanged(golden):
    """The W&B key mapping is untouched by the rewrite; pin it anyway."""
    groups = _metric_groups(
        "val",
        {"loss/siglip": 0.15, "loss/ts_ecg": 0.16, "loss/distill": 0.1, "loss/total": 0.25},
        {
            "n/img_image": 2,
            "n/img_video": 1,
            "n/ts_signal": 9,
            "n/ts_smellnet": 3,
            "n/ts_ecg": 3,
            "n/ts_tactile": 3,
            "n/skipped_image": 1,
            "n/skipped_video": 0,
            "n/skipped_signal": 0,
        },
        golden["golden"]["mixed"],
    )
    _assert_same(groups, golden["golden"]["groups_val"], "groups_val")


def test_streaming_equals_one_shot(golden):
    """Three microbatches must produce exactly the whole-batch numbers.

    This is the property the rewrite trades embedding retention for: because every
    statistic is a sum over rows, folding rows in as they arrive has to be
    indistinguishable from ranking them all at once. Macro-F1 is the sharp case --
    it is a pooled-confusion number, so an implementation that averaged
    per-microbatch F1s would diverge here.
    """
    data = golden["inputs"]["ecg"]
    spec = _single_label_spec(data["candidates"], data["bank"])
    z = _tensor(data["z"])

    stats = new_stats((spec,), torch.device("cpu"))
    for start in range(0, len(z), 11):
        chunk = slice(start, start + 11)
        update_stats(
            stats,
            (spec,),
            (z[chunk], data["labels"][chunk], ["ecg"] * len(z[chunk]), None, None),
            None,
            None,
        )
    streamed = prediction_metrics(stats, (spec,))

    expected = {f"{key}/ts_ecg": value for key, value in golden["golden"]["ecg"].items()}
    expected.update({f"{key}/overall": value for key, value in golden["golden"]["ecg"].items()})
    _assert_same(streamed, expected, "streamed")
