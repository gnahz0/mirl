# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Sensor-to-text ranking metrics and their single distributed reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import MULTILABEL_TASKS, TACTILE_SPANS, TASK_LABELS

# Metric reduction order must be identical on every rank.
_TS_FAMILIES: tuple[str, ...] = ("smellnet", "ecg", "tactile")
_CLASSIFICATION_FAMILIES = ("smellnet", "ecg")
_PUBLIC_STATS = ("accuracy", "f1_macro", "recall_at_1", "recall_at_5", "map")
_ALL_STATS = (*_PUBLIC_STATS, "prediction_coverage")

_REDUCED_METRIC_KEYS = (
    "loss/siglip",
    "loss/ts_smellnet",
    "loss/ts_ecg",
    "loss/ts_tactile",
    *(f"loss/task/{task}" for task in TASK_LABELS),
    "loss/distill",
    "loss/total",
)

# Derive n/ts_signal after reducing its family counts.
_COUNT_KEYS: tuple[str, ...] = (
    "n/img_image",
    "n/img_video",
    *(f"n/ts_{family}" for family in _TS_FAMILIES),
    *(f"n/skipped_{kind}" for kind in ("image", "video", "signal")),
)

# Every published statistic is a ratio of sums over rows, so these nine tensors
# are all any reduction ever needs. Per-class counts are one entry per label;
# the rest are scalars.
_CLASS_FIELDS = ("support", "predicted", "true_positive", "recall_at_5_count")
_SCALAR_FIELDS = ("top_is_positive", "recall_at_1", "recall_at_5", "average_precision", "n")


def all_reduce_sum(values: dict[str, torch.Tensor], *, world_size: int = 1) -> dict[str, torch.Tensor]:
    """Sum every entry of ``values`` across ranks with exactly one collective.

    Callers build ``values`` from a key set fixed before any data is seen, so the
    packed buffer is identical on every rank whatever rows landed locally.
    """
    if world_size <= 1:
        return values
    keys = sorted(values)
    flat = torch.cat([values[key].reshape(-1).double() for key in keys])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    reduced: dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        size = values[key].numel()
        reduced[key] = flat[offset : offset + size].view(values[key].shape)
        offset += size
    return reduced


@dataclass(frozen=True)
class BankSpec:
    """One scoring unit: some rows ranked against some bank of label embeddings.

    The only thing that differs between smellnet, ecg and the six tactile tasks,
    and it is data rather than a code path -- all eight score through
    ``_bank_stats``. ``label_to_id`` selects rows by batch family and one-hots the
    ground-truth string; ``span`` selects them by the tactile mask and slices that
    task's columns. ``threshold_bias`` predicts every label whose biased logit
    exceeds zero, else the prediction is the argmax within this bank.
    """

    key: str  # "ts_smellnet" | "task/force_level"
    labels: tuple[str, ...]
    embeddings: torch.Tensor  # (K, D) frozen bank or bank slice
    family: str | None = None
    label_to_id: dict[str, int] | None = None
    threshold_bias: torch.Tensor | None = None  # (K,)
    span: tuple[int, int] | None = None
    report_key: str | None = None  # emit per-class W&B rows under this name
    row_extra: dict[str, object] | None = None


def build_bank_specs(label_bank: dict, tactile_bank: tuple | None) -> tuple[BankSpec, ...]:
    """Enumerate every scoring unit, once, before any data is seen.

    Depends only on the label banks and ``TACTILE_SPANS``, so the unit list -- and
    the reduce buffer's length -- is identical on every rank for the whole run.
    """
    specs: list[BankSpec] = []
    for family in _TS_FAMILIES:
        entry = label_bank.get(family)
        if entry is None:
            continue
        labels, embeddings = entry
        specs.append(
            BankSpec(
                key=f"ts_{family}",
                labels=labels,
                embeddings=embeddings,
                family=family,
                label_to_id={label: index for index, label in enumerate(labels)},
                report_key=family if family in _CLASSIFICATION_FAMILIES else None,
            )
        )
    if tactile_bank is not None:
        all_labels, embeddings, bias = tactile_bank
        for task, (start, stop) in TACTILE_SPANS.items():
            specs.append(
                BankSpec(
                    key=f"task/{task}",
                    labels=all_labels[start:stop],
                    embeddings=embeddings[start:stop],
                    threshold_bias=bias[start:stop] if task in MULTILABEL_TASKS else None,
                    span=(start, stop),
                    row_extra={"task": task},
                )
            )
    return tuple(specs)


@torch.no_grad()
def _bank_stats(
    z: torch.Tensor,
    target: torch.Tensor,
    spec: BankSpec,
    log_logit_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sufficient statistics for one (rows x bank) scoring problem. No collective.

    Single-label units pass a one-hot target and every statistic collapses to its
    familiar form: average precision over one positive is ``1 / rank``, recall@1 is
    accuracy. These sums add across microbatches exactly as they add across ranks,
    so nothing retains embeddings and macro-F1 stays a pooled-confusion number.
    Zero rows is legal and yields zeros, which keeps an empty rank in lockstep.
    """
    num_labels = len(spec.labels)
    target = target.to(z.device).bool()
    sims = z.float() @ spec.embeddings.to(z.device).float().t()
    # stable=True pins tie order to label index, so a row scores identically
    # however the rows were chunked.
    ranked = sims.argsort(dim=1, descending=True, stable=True)
    top = ranked[:, 0]
    top_k = ranked[:, : min(5, num_labels)]

    top_is_positive = target.gather(1, top[:, None]).squeeze(1)
    positive_count = target.sum(dim=1).clamp_min(1)
    recall_at_1 = top_is_positive.float() / positive_count
    recall_at_5 = target.gather(1, top_k).sum(dim=1) / positive_count

    ranked_target = target.gather(1, ranked).float()
    ranks = torch.arange(1, num_labels + 1, device=z.device, dtype=torch.float32)
    precision_at_rank = ranked_target.cumsum(dim=1) / ranks
    average_precision = (precision_at_rank * ranked_target).sum(dim=1) / ranked_target.sum(
        dim=1
    ).clamp_min(1)

    if spec.threshold_bias is None:
        predicted_mask = F.one_hot(top, num_classes=num_labels).bool()
    else:
        logits = log_logit_scale.float().exp() * sims + spec.threshold_bias.to(z.device)
        predicted_mask = logits > 0
    in_top_k = torch.zeros_like(target)
    in_top_k.scatter_(1, top_k, True)

    return {
        "support": target.sum(dim=0).double(),
        "predicted": predicted_mask.sum(dim=0).double(),
        "true_positive": (target & predicted_mask).sum(dim=0).double(),
        "recall_at_5_count": (target & in_top_k).sum(dim=0).double(),
        "top_is_positive": top_is_positive.double().sum(),
        "recall_at_1": recall_at_1.double().sum(),
        "recall_at_5": recall_at_5.double().sum(),
        "average_precision": average_precision.double().sum(),
        "n": z.new_tensor(float(len(target)), dtype=torch.float64),
    }


@torch.no_grad()
def _bank_metrics(
    stats: dict[str, torch.Tensor],
    spec: BankSpec,
    *,
    rows_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Turn globally reduced sufficient statistics into the published scalars."""
    support, predicted = stats["support"], stats["predicted"]
    true_positive, sample_count = stats["true_positive"], stats["n"]

    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    recall_at_5_by_class = stats["recall_at_5_count"] / support.clamp_min(1)
    supported = support > 0

    if rows_out is not None:
        rows_out.extend(
            {
                **(spec.row_extra or {}),
                "class_id": index,
                "label": label,
                "support": support[index].item(),
                "predicted": predicted[index].item(),
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "recall_at_5": recall_at_5_by_class[index].item(),
            }
            for index, label in enumerate(spec.labels)
        )

    return {
        "accuracy": float(stats["top_is_positive"] / sample_count),
        "f1_macro": float(f1[supported].mean()),
        "recall_at_1": float(stats["recall_at_1"] / sample_count),
        "recall_at_5": float(stats["recall_at_5"] / sample_count),
        "map": float(stats["average_precision"] / sample_count),
        "prediction_coverage": float((predicted > 0).sum()) / len(spec.labels),
    }


@torch.no_grad()
def score_bank(
    spec: BankSpec,
    z: torch.Tensor,
    target: torch.Tensor,
    *,
    log_logit_scale: torch.Tensor | None = None,
    rows_out: list[dict[str, object]] | None = None,
    world_size: int = 1,
) -> dict[str, float]:
    """Score one unit in one shot: accumulate, reduce, derive."""
    stats = all_reduce_sum(_bank_stats(z, target, spec, log_logit_scale), world_size=world_size)
    return _bank_metrics(stats, spec, rows_out=rows_out)


def new_stats(specs: tuple[BankSpec, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Zero-fill the complete statistic key set before any data is seen."""
    stats: dict[str, torch.Tensor] = {}
    for spec in specs:
        for field_name in _CLASS_FIELDS:
            stats[f"{spec.key}/{field_name}"] = torch.zeros(
                len(spec.labels), dtype=torch.float64, device=device
            )
        for field_name in _SCALAR_FIELDS:
            stats[f"{spec.key}/{field_name}"] = torch.zeros((), dtype=torch.float64, device=device)
    return stats


def _unit_rows(
    spec: BankSpec,
    z: torch.Tensor,
    texts: list[str],
    families: list[str],
    targets: torch.Tensor | None,
    masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select this unit's rows and multi-hot target from one microbatch."""
    if spec.span is None:
        index = [i for i, value in enumerate(families) if value == spec.family]
        if not index:
            return None
        selected = torch.tensor(index, device=z.device, dtype=torch.long)
        ids = torch.tensor([spec.label_to_id[texts[i]] for i in index], device=z.device)
        return z[selected], F.one_hot(ids, num_classes=len(spec.labels))
    if masks is None:
        return None
    start, stop = spec.span
    span_mask = masks[:, start:stop] > 0
    rows = span_mask[:, 0]
    # collate_alignment writes a task's mask columns as one slice; that is the only
    # reason one column recovers the per-row mask, and nothing in data.py enforces it.
    if not torch.equal(span_mask, rows[:, None].expand_as(span_mask)):
        raise ValueError(f"tactile mask is not span-uniform for {spec.key}")
    return z[rows], targets[rows, start:stop]


@torch.no_grad()
def update_stats(
    stats: dict[str, torch.Tensor],
    specs: tuple[BankSpec, ...],
    ts_eval: tuple,
    log_logit_scale: torch.Tensor | None,
) -> None:
    """Fold one microbatch into the running statistics. No collective, no retention."""
    z, texts, families, targets, masks = ts_eval
    if z is None:
        return
    for spec in specs:
        rows = _unit_rows(spec, z, texts, families, targets, masks)
        if rows is None:
            continue
        for field_name, value in _bank_stats(rows[0], rows[1], spec, log_logit_scale).items():
            # Fixed key set: a diverged rank raises KeyError rather than hanging.
            key = f"{spec.key}/{field_name}"
            stats[key] = stats[key] + value


@torch.no_grad()
def prediction_metrics(
    stats: dict[str, torch.Tensor],
    specs: tuple[BankSpec, ...],
    *,
    per_class: dict[str, list[dict[str, object]]] | None = None,
    per_label: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Derive every published key from globally reduced statistics. No collective."""
    metrics: dict[str, float] = {}
    task_scores: list[dict[str, float]] = []
    for spec in specs:
        # Reads an already-reduced count, so every rank takes the same branch.
        unit = {name: stats[f"{spec.key}/{name}"] for name in (*_CLASS_FIELDS, *_SCALAR_FIELDS)}
        if not float(unit["n"]):
            continue
        rows_out = None
        if spec.report_key is not None and per_class is not None:
            rows_out = per_class.setdefault(spec.report_key, [])
        elif spec.span is not None and per_label is not None:
            rows_out = per_label
        scores = _bank_metrics(unit, spec, rows_out=rows_out)
        for stat in _ALL_STATS:
            metrics[f"{stat}/{spec.key}"] = scores[stat]
        if spec.span is not None:
            task_scores.append(scores)
    if task_scores:
        for stat in _ALL_STATS:
            metrics[f"{stat}/ts_tactile"] = sum(s[stat] for s in task_scores) / len(task_scores)
    return _merge_prediction_metrics(metrics)


def _merge_prediction_metrics(*metric_sets: dict[str, float]) -> dict[str, float]:
    """Merge family metrics and recompute an equal-family overall score."""
    merged: dict[str, float] = {}
    for values in metric_sets:
        merged.update(values)
    for stat in _PUBLIC_STATS:
        family_values = [merged[f"{stat}/ts_{family}"] for family in _TS_FAMILIES if f"{stat}/ts_{family}" in merged]
        if family_values:
            merged[f"{stat}/overall"] = sum(family_values) / len(family_values)
        else:
            merged.pop(f"{stat}/overall", None)
    return merged


def _metric_groups(
    split: str,
    loss_metrics: dict[str, float],
    counts: dict[str, int],
    prediction_metrics: dict[str, float],
) -> dict[str, float]:
    """Build the shared train/validation W&B metric surface."""
    core = f"{split}-core"
    aux = f"{split}-aux"
    out: dict[str, float] = {f"{core}/loss/aggregate": loss_metrics["loss/total"]}

    skipped = {kind: counts.get(f"n/skipped_{kind}", 0) for kind in ("image", "video", "signal")}
    skipped_total = sum(skipped.values())
    valid_total = (
        counts.get("n/img_image", 0)
        + counts.get("n/img_video", 0)
        + sum(counts.get(f"n/ts_{family}", 0) for family in _TS_FAMILIES)
    )
    for kind, value in skipped.items():
        out[f"{aux}/n/skipped/{kind}"] = float(value)
    out[f"{aux}/n/skipped/total"] = float(skipped_total)
    out[f"{aux}/skipped_fraction"] = skipped_total / max(valid_total + skipped_total, 1)

    for name in ("siglip", "distill"):
        key = f"loss/{name}"
        if key in loss_metrics:
            out[f"{aux}/loss/{name}"] = loss_metrics[key]

    for family in _TS_FAMILIES:
        loss_key = f"loss/ts_{family}"
        if loss_key in loss_metrics:
            out[f"{core}/loss/{family}"] = loss_metrics[loss_key]
        for stat in _PUBLIC_STATS:
            key = f"{stat}/ts_{family}"
            if key in prediction_metrics:
                out[f"{core}/{stat}/{family}"] = prediction_metrics[key]
        coverage_key = f"prediction_coverage/ts_{family}"
        if coverage_key in prediction_metrics:
            out[f"{aux}/prediction_coverage/{family}"] = prediction_metrics[coverage_key]
        out[f"{aux}/n/{family}"] = float(counts[f"n/ts_{family}"])

    for task in TASK_LABELS:
        loss_key = f"loss/task/{task}"
        if loss_key in loss_metrics:
            out[f"{aux}/loss/tactile/{task}"] = loss_metrics[loss_key]
        for stat in _ALL_STATS:
            key = f"{stat}/task/{task}"
            if key in prediction_metrics:
                out[f"{aux}/{stat}/tactile/{task}"] = prediction_metrics[key]

    for stat in _PUBLIC_STATS:
        key = f"{stat}/overall"
        if key in prediction_metrics:
            out[f"{core}/{stat}/overall"] = prediction_metrics[key]

    if split == "train":
        for key, value in loss_metrics.items():
            if key.startswith("grad_norm/") or key in {"grad_norm", "logit_scale"}:
                out[f"{aux}/{key}"] = value
        out[f"{aux}/n/img"] = float(counts["n/img_image"] + counts["n/img_video"])

    return out
