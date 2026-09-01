"""Guard the metric surface established by the 2026-08-09 rewrite.

``fixtures/metrics_golden.json`` stores inputs and outputs produced by
``regen_metrics_golden.py`` against the pre-rewrite implementation; these tests
replay the same inputs through the new unified path and compare exactly, except
tactile thresholded coverage, which now uses SigLIP's learned scalar bias
instead of fixed per-task priors (ranking metrics are unchanged by that).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mirl_ext.alignment.data import TACTILE_SPANS  # noqa: E402
from mirl_ext.data.schema import MULTILABEL_TASKS  # noqa: E402
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
    # Prediction coverage is a later additive diagnostic (tested in test_losses.py), absent from the fixture.
    actual = {key: value for key, value in actual.items() if "prediction_coverage" not in key}
    assert set(actual) == set(expected), f"{label}: key set changed"
    for key, value in expected.items():
        assert actual[key] == value, f"{label}: {key} moved {value!r} -> {actual[key]!r}"


def _assert_rows(actual: list[dict], expected: list[dict], label: str) -> None:
    assert len(actual) == len(expected), f"{label}: row count changed"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert got == want, f"{label}: row {index} changed"


@pytest.mark.parametrize("case", ["smell", "ecg"])
def test_single_label_families_score_unchanged(golden, case):
    """Both golden vectors keep every scalar and per-class row. The "smell" data
    survives smellnet's exclusion as a pure numeric vector (50 labels, skewed
    support, unobserved classes) scored through a generically keyed spec."""
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
    """Changing calibration leaves ranking intact and changes thresholded predictions."""
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

    # prediction_metrics now emits /overall itself; with only tactile present each equals its ts_tactile value.
    expected = dict(golden["golden"]["tactile"])
    expected.update(
        {
            f"{stat}/overall": expected[f"{stat}/ts_tactile"]
            for stat in ("accuracy", "recall_at_1", "recall_at_5", "map")
        }
    )
    _assert_same(metrics, expected, "tactile ranking")

    expected_rows = golden["golden"]["tactile_rows"]
    for actual, old in zip(rows, expected_rows, strict=True):
        for field in ("task", "class_id", "label", "support", "recall_at_5"):
            assert actual[field] == old[field]
        if actual["task"] in MULTILABEL_TASKS:
            assert actual["predicted"] == actual["precision"] == actual["recall"] == 0
        else:
            assert actual == old
    # local_shape was answered by no row, so it must be absent rather than nan.
    assert not any(key.endswith("/task/local_shape") for key in metrics)
    assert {row["task"] for row in rows} == set(TACTILE_SPANS) - {"local_shape"}


def test_mixed_families_and_overall_rollup_unchanged(golden):
    """ecg scored alongside golden tactile, then the equal-family overall.
    (smellnet left _TS_FAMILIES 2026-08-31; the overall mean is recomputed here
    from the per-family golden values instead of pinned as a scalar.)"""
    ecg = golden["inputs"]["ecg"]
    specs = build_bank_specs({"ecg": (tuple(ecg["candidates"]), _tensor(ecg["bank"]))}, None)
    reports: dict[str, list[dict]] = {}
    stats = new_stats(specs, torch.device("cpu"))
    update_stats(
        stats,
        specs,
        (_tensor(ecg["z"]), list(ecg["labels"]), ["ecg"] * len(ecg["labels"]), None, None),
        None,
        None,
    )
    mixed = _merge_prediction_metrics(prediction_metrics(stats, specs, per_class=reports), golden["golden"]["tactile"])

    expected = {f"{key}/ts_ecg": value for key, value in golden["golden"]["ecg"].items()}
    expected.update(golden["golden"]["tactile"])
    for stat in ("accuracy", "recall_at_1", "recall_at_5", "map"):
        expected[f"{stat}/overall"] = (
            golden["golden"]["ecg"][stat] + golden["golden"]["tactile"][f"{stat}/ts_tactile"]
        ) / 2
    _assert_same(mixed, expected, "mixed")
    assert sorted(reports) == ["ecg"]


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
        },
        golden["golden"]["mixed"],
    )
    expected = {key: value for key, value in golden["golden"]["groups_val"].items() if "smellnet" not in key}
    _assert_same(groups, expected, "groups_val")


def test_streaming_equals_one_shot(golden):
    """Streaming microbatches must exactly match the one-shot numbers: every
    statistic is a sum over rows, the property that replaced embedding retention."""
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
