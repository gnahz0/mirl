"""Record the metric surface produced by the PRE-REFACTOR implementation.

Ran once before the 2026-08-09 metrics rewrite to produce
``fixtures/metrics_golden.json``; the functions it calls no longer exist. Every
block also reproduces from ``0e1f4272`` except ``tactile_rows`` (that tree already
carried the ``_score_against_bank`` merge) -- treat that block as a lock-forward
pin. Do NOT regenerate against current code: the golden test would then only
assert that the code agrees with itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mirl_ext.alignment.data import TACTILE_NUM_LABELS, TACTILE_SPANS

try:
    from mirl_ext.alignment.metrics import (
        _label_ranking_metrics,
        _merge_prediction_metrics,
        _metric_groups,
        _tactile_prediction_metrics,
        _ts_prediction_metrics,
    )
except ImportError as error:  # pragma: no cover - provenance script, not a test
    raise SystemExit(
        f"{error}\n\n"
        "This generator targets the pre-2026-08-09 metrics API and cannot run "
        "against the current one. See this module's docstring; regenerating the "
        "fixture from today's code would destroy the comparison it exists for."
    ) from error

FIXTURE = Path(__file__).parent / "fixtures" / "metrics_golden.json"


def _unit(generator: torch.Generator, rows: int, dim: int = 64) -> torch.Tensor:
    return F.normalize(torch.randn(rows, dim, generator=generator), dim=-1)


def _inputs() -> dict:
    """Fixed random inputs, stored alongside the outputs they produced."""
    generator = torch.Generator().manual_seed(0)

    # smellnet: many labels, skewed support, several classes never observed.
    smell_labels = tuple(f"smell_{index:02d}" for index in range(50))
    smell_draw = torch.randint(0, 12, (24,), generator=generator).tolist()

    # ecg: few labels, a dominant majority class, one class entirely absent.
    ecg_labels = ("normal", "af", "lbbb", "rbbb", "pvc", "pac", "st")
    ecg_draw = [0] * 14 + [1] * 6 + [2] * 4 + [3] * 4 + [4] * 3 + [5] * 1

    # tactile: every task observed except local_shape; row 15 answers only initial_fingers (incomplete join).
    targets = torch.zeros(16, TACTILE_NUM_LABELS)
    masks = torch.zeros(16, TACTILE_NUM_LABELS)
    for row in range(16):
        for task, (start, stop) in TACTILE_SPANS.items():
            if task == "local_shape":
                continue
            if row == 15 and task != "initial_fingers":
                continue
            masks[row, start:stop] = 1.0
            width = stop - start
            picked = int(torch.randint(0, width, (1,), generator=generator))
            targets[row, start + picked] = 1.0
            if task == "initial_fingers" and row % 3 == 0:
                targets[row, start + (picked + 1) % width] = 1.0

    rates = {task: 0.15 + 0.05 * index for index, task in enumerate(TACTILE_SPANS)}
    bias = torch.tensor(
        [
            float(torch.log(torch.tensor(rates[task] / (1.0 - rates[task]))))
            for task, (start, stop) in TACTILE_SPANS.items()
            for _ in range(stop - start)
        ]
    )

    return {
        "smell": {
            "z": _unit(generator, 24),
            "labels": [smell_labels[index] for index in smell_draw],
            "candidates": smell_labels,
            "bank": _unit(generator, 50),
        },
        "ecg": {
            "z": _unit(generator, 32),
            "labels": [ecg_labels[index] for index in ecg_draw],
            "candidates": ecg_labels,
            "bank": _unit(generator, 7),
        },
        "tactile": {
            "z": _unit(generator, 16),
            "labels": tuple(f"tactile_{index:02d}" for index in range(TACTILE_NUM_LABELS)),
            "bank": _unit(generator, TACTILE_NUM_LABELS),
            "targets": targets,
            "masks": masks,
            "bias": bias,
            "log_logit_scale": float(torch.log(torch.tensor(1.0 / 0.07))),
        },
    }


def _rows(report: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items()} for row in report]


def main() -> None:
    data = _inputs()
    smell, ecg, tactile = data["smell"], data["ecg"], data["tactile"]
    golden: dict[str, object] = {}

    smell_rows: list[dict] = []
    golden["smellnet"] = _label_ranking_metrics(
        smell["z"], smell["labels"], smell["candidates"], smell["bank"], per_class_out=smell_rows
    )
    golden["smellnet_rows"] = _rows(smell_rows)

    ecg_rows: list[dict] = []
    golden["ecg"] = _label_ranking_metrics(
        ecg["z"], ecg["labels"], ecg["candidates"], ecg["bank"], per_class_out=ecg_rows
    )
    golden["ecg_rows"] = _rows(ecg_rows)

    tactile_rows: list[dict] = []
    golden["tactile"] = _tactile_prediction_metrics(
        tactile["z"],
        tactile["targets"],
        tactile["masks"],
        (tactile["labels"], tactile["bank"], tactile["bias"]),
        torch.tensor(tactile["log_logit_scale"]),
        per_label_out=tactile_rows,
    )
    golden["tactile_rows"] = _rows(tactile_rows)

    # Both single-label families through one call, then the equal-family rollup.
    mixed_reports: dict[str, list[dict]] = {}
    mixed = _ts_prediction_metrics(
        torch.cat([smell["z"], ecg["z"]]),
        [*smell["labels"], *ecg["labels"]],
        ["smellnet"] * 24 + ["ecg"] * 32,
        {
            "smellnet": (smell["candidates"], smell["bank"]),
            "ecg": (ecg["candidates"], ecg["bank"]),
        },
        per_class_reports=mixed_reports,
    )
    golden["mixed"] = _merge_prediction_metrics(mixed, golden["tactile"])
    golden["mixed_report_families"] = sorted(mixed_reports)

    golden["groups_val"] = _metric_groups(
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
        golden["mixed"],
    )

    payload = {
        "inputs": {
            "smell": {
                "z": smell["z"].tolist(),
                "labels": smell["labels"],
                "candidates": list(smell["candidates"]),
                "bank": smell["bank"].tolist(),
            },
            "ecg": {
                "z": ecg["z"].tolist(),
                "labels": ecg["labels"],
                "candidates": list(ecg["candidates"]),
                "bank": ecg["bank"].tolist(),
            },
            "tactile": {
                "z": tactile["z"].tolist(),
                "labels": list(tactile["labels"]),
                "bank": tactile["bank"].tolist(),
                "targets": tactile["targets"].tolist(),
                "masks": tactile["masks"].tolist(),
                "bias": tactile["bias"].tolist(),
                "log_logit_scale": tactile["log_logit_scale"],
            },
        },
        "golden": golden,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    for name in ("smellnet", "ecg", "tactile", "mixed"):
        print(f"  {name}: {len(golden[name])} scalars")


if __name__ == "__main__":
    main()
