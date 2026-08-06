# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Sensor-to-text ranking metrics and distributed loss reduction."""

from __future__ import annotations

import torch
import torch.distributed as dist

# Metric reduction order must be identical on every rank.
_TS_FAMILIES: tuple[str, ...] = ("smellnet", "ecg", "tactile")
_CLASSIFICATION_FAMILIES = ("smellnet", "ecg")
_CLASSIFICATION_STATS = ("accuracy", "f1_macro", "recall_at_5")
_RETRIEVAL_STATS = ("recall_at_1", "recall_at_5", "map")
_PUBLIC_STATS = ("accuracy", "f1_macro", "recall_at_1", "recall_at_5", "map")

_REDUCED_METRIC_KEYS = (
    "loss/siglip",
    "loss/ts_smellnet",
    "loss/ts_ecg",
    "loss/ts_tactile",
    "loss/distill",
    "loss/total",
)

_COUNT_KEYS: tuple[str, ...] = (
    "n/img_image",
    "n/img_video",
    "n/ts_signal",
) + tuple(f"n/ts_{family}" for family in _TS_FAMILIES)

def new_counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _allreduce_metrics(metrics: dict, device: torch.device, world_size: int) -> dict:
    """Average each present loss over the ranks that computed it."""
    missing = [key for key in metrics if key.startswith("loss/") and key not in _REDUCED_METRIC_KEYS]
    if missing:
        raise RuntimeError(f"metrics absent from _REDUCED_METRIC_KEYS: {missing}")
    if world_size <= 1:
        return metrics
    # Pack once to avoid a device synchronization per metric.
    flat: list[float] = []
    for key in _REDUCED_METRIC_KEYS:
        value = metrics.get(key)
        flat += [float(value), 1.0] if value is not None else [0.0, 0.0]
    packed = torch.tensor(flat, device=device, dtype=torch.float64)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.tolist()
    out = dict(metrics)
    for i, key in enumerate(_REDUCED_METRIC_KEYS):
        total, present = reduced[2 * i], reduced[2 * i + 1]
        if present > 0:
            out[key] = total / present
        else:
            out.pop(key, None)
    return out


def _allreduce_counts(counts: dict, device: torch.device, world_size: int) -> dict:
    """Sum sample counts across ranks in a fixed collective order."""
    if world_size <= 1:
        return counts
    packed = torch.tensor([float(counts.get(k, 0)) for k in _COUNT_KEYS], device=device, dtype=torch.float64)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {k: int(v) for k, v in zip(_COUNT_KEYS, packed.tolist(), strict=True)}


def add_ts_family_counts(counts: dict, batch: dict) -> None:
    """Add per-family row counts."""
    for family in batch.get("ts_format") or []:
        counts[f"n/ts_{family}"] += 1


@torch.no_grad()
def _effective_dim(z: torch.Tensor) -> float | None:
    """Participation ratio of the centered embedding covariance spectrum."""
    if z.shape[0] < 3:
        return None
    centered = z.float() - z.float().mean(dim=0, keepdim=True)
    ev = torch.linalg.svdvals(centered) ** 2
    total = float(ev.sum())
    if total <= 0.0:
        return None
    return float((total**2) / float((ev**2).sum()))


@torch.no_grad()
def _label_ranking_metrics(
    z: torch.Tensor,
    labels: list[str],
    candidate_labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    per_class_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Score labels, macro-averaging only classes present in ground truth."""
    label_to_id = {label: idx for idx, label in enumerate(candidate_labels)}
    text_embeddings = text_embeddings.to(device=z.device)
    true = torch.tensor([label_to_id[label] for label in labels], device=z.device)
    sims = z.float() @ text_embeddings.float().t()
    ranked = sims.argsort(dim=1, descending=True)
    pred = ranked[:, 0]
    recall_at_5 = (ranked[:, : min(5, ranked.shape[1])] == true[:, None]).any(dim=1)
    reciprocal_rank = (ranked == true[:, None]).int().argmax(dim=1).add(1).float().reciprocal()
    num_classes = len(candidate_labels)
    support = torch.bincount(true, minlength=num_classes)
    predicted = torch.bincount(pred, minlength=num_classes)
    true_positive = torch.bincount(true[pred == true], minlength=num_classes)

    support_f = support.float()
    predicted_f = predicted.float()
    precision = true_positive.float() / predicted_f.clamp_min(1)
    recall = true_positive.float() / support_f.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    recall_at_5_by_class = torch.bincount(
        true,
        weights=recall_at_5.float(),
        minlength=num_classes,
    ) / support_f.clamp_min(1)
    supported = support > 0

    if per_class_out is not None:
        per_class_out.extend(
            {
                "class_id": class_id,
                "label": label,
                "support": support_i,
                "predicted": predicted_i,
                "precision": precision_i,
                "recall": recall_i,
                "f1": f1_i,
                "recall_at_5": recall_at_5_i,
            }
            for class_id, (label, support_i, predicted_i, precision_i, recall_i, f1_i, recall_at_5_i) in enumerate(
                zip(
                    candidate_labels,
                    support.tolist(),
                    predicted.tolist(),
                    precision.tolist(),
                    recall.tolist(),
                    f1.tolist(),
                    recall_at_5_by_class.tolist(),
                    strict=True,
                )
            )
        )

    return {
        "accuracy": float((pred == true).float().mean()),
        "f1_macro": float(f1[supported].mean()),
        "recall_at_1": float((pred == true).float().mean()),
        "recall_at_5": float(recall_at_5.float().mean()),
        "map": float(reciprocal_rank.mean()),
        "prediction_coverage": float((predicted > 0).sum()) / num_classes,
    }


@torch.no_grad()
def _ts_prediction_metrics(
    z: torch.Tensor,
    texts: list[str],
    families: list[str],
    label_bank: dict[str, tuple[tuple[str, ...], torch.Tensor]],
    per_class_reports: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, float]:
    """Compute honest per-family metrics and an equal-family overall score."""
    metrics: dict[str, float] = {}
    family_scores: dict[str, dict[str, float]] = {}
    for family in _TS_FAMILIES:
        idx = [i for i, value in enumerate(families) if value == family]
        if not idx:
            continue
        sel = torch.tensor(idx, device=z.device, dtype=torch.long)
        family_labels = [texts[i] for i in idx]
        entry = label_bank[family]
        report_rows = (
            per_class_reports.setdefault(family, [])
            if per_class_reports is not None and family in _CLASSIFICATION_FAMILIES
            else None
        )
        family_metrics = _label_ranking_metrics(
            z[sel],
            family_labels,
            entry[0],
            entry[1],
            per_class_out=report_rows,
        )
        if family in _CLASSIFICATION_FAMILIES:
            family_scores[family] = family_metrics
            published = (*_CLASSIFICATION_STATS, "prediction_coverage")
        else:
            published = _RETRIEVAL_STATS
        for stat in published:
            metrics[f"{stat}/ts_{family}"] = family_metrics[stat]
        fam_eff = _effective_dim(z[sel])
        if fam_eff is not None:
            metrics[f"eff_dim/ts_{family}"] = fam_eff

    for stat in ("accuracy", "f1_macro"):
        values = [scores[stat] for scores in family_scores.values() if stat in scores]
        if values:
            metrics[f"{stat}/overall"] = sum(values) / len(values)

    return metrics


def _publish_family_metrics(
    out: dict[str, float],
    family: str,
    loss_metrics: dict[str, float],
    prediction_metrics: dict[str, float],
    *,
    core_prefix: str,
    aux_prefix: str,
) -> None:
    loss_key = f"loss/ts_{family}"
    if loss_key in loss_metrics:
        out[f"{core_prefix}/loss/{family}"] = loss_metrics[loss_key]

    for stat in _PUBLIC_STATS:
        key = f"{stat}/ts_{family}"
        if key in prediction_metrics:
            out[f"{core_prefix}/{stat}/{family}"] = prediction_metrics[key]
    for source, target in (("eff_dim", "effective_dimension"), ("prediction_coverage", "prediction_coverage")):
        key = f"{source}/ts_{family}"
        if key in prediction_metrics:
            out[f"{aux_prefix}/{target}/{family}"] = prediction_metrics[key]


def _publish_supervised_aggregates(
    out: dict[str, float],
    metrics: dict[str, float],
    core_prefix: str,
) -> None:
    for stat in ("accuracy", "f1_macro"):
        key = f"{stat}/overall"
        if key in metrics:
            out[f"{core_prefix}/{stat}/overall"] = metrics[key]


def _training_metric_groups(
    metrics: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    """Return the compact, public training metric surface for W&B."""
    out: dict[str, float] = {"train-core/loss/aggregate": metrics["loss/total"]}
    for name in ("siglip", "distill"):
        if f"loss/{name}" in metrics:
            out[f"train-aux/loss/{name}"] = metrics[f"loss/{name}"]
    for family in _TS_FAMILIES:
        _publish_family_metrics(
            out,
            family,
            metrics,
            metrics,
            core_prefix="train-core",
            aux_prefix="train-aux",
        )
        out[f"train-aux/n/{family}"] = float(counts[f"n/ts_{family}"])

    _publish_supervised_aggregates(out, metrics, "train-core")

    for key, value in metrics.items():
        if key.startswith("grad_norm/") or key in {"grad_norm", "logit_scale"}:
            out[f"train-aux/{key}"] = value
    out["train-aux/n/img"] = float(counts["n/img_image"] + counts["n/img_video"])
    return out


def _validation_metric_groups(
    averaged_metrics: dict[str, float],
    prediction_metrics: dict[str, float],
    bucket_totals: dict[str, int],
) -> dict[str, float]:
    """Return core selection metrics and a small set of validation diagnostics."""
    out = {"val-core/loss/aggregate": averaged_metrics["loss/total"]}
    for name in ("siglip", "distill"):
        if f"loss/{name}" in averaged_metrics:
            out[f"val-aux/loss/{name}"] = averaged_metrics[f"loss/{name}"]
    for family in _TS_FAMILIES:
        _publish_family_metrics(
            out,
            family,
            averaged_metrics,
            prediction_metrics,
            core_prefix="val-core",
            aux_prefix="val-aux",
        )
        out[f"val-aux/n/{family}"] = float(bucket_totals[f"n/ts_{family}"])

    _publish_supervised_aggregates(out, prediction_metrics, "val-core")

    return out
