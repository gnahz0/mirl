# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Sensor-to-text ranking metrics and distributed loss reduction."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import MULTILABEL_TASKS, TASK_LABELS

# Metric reduction order must be identical on every rank.
_TS_FAMILIES: tuple[str, ...] = ("smellnet", "ecg", "tactile")
_CLASSIFICATION_FAMILIES = ("smellnet", "ecg")
_PUBLIC_STATS = ("accuracy", "f1_macro", "recall_at_1", "recall_at_5", "map")

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
    (
        "n/img_image",
        "n/img_video",
    )
    + tuple(f"n/ts_{family}" for family in _TS_FAMILIES)
    + tuple(f"n/skipped_{kind}" for kind in ("image", "video", "signal"))
)


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
        out = {key: int(counts.get(key, 0)) for key in _COUNT_KEYS}
    else:
        packed = torch.tensor([counts.get(key, 0) for key in _COUNT_KEYS], device=device, dtype=torch.long)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        out = {k: int(v) for k, v in zip(_COUNT_KEYS, packed.tolist(), strict=True)}
    out["n/ts_signal"] = sum(out[f"n/ts_{family}"] for family in _TS_FAMILIES)
    return out


def add_batch_counts(counts: dict, batch: dict) -> None:
    """Count one source-homogeneous local batch."""
    for kind, value in batch.get("skipped", {}).items():
        if value:
            counts[f"n/skipped_{kind}"] += int(value)
    size = len(batch["media"])
    if batch["kind"] == "signal":
        counts[f"n/ts_{batch['family']}"] += size
    else:
        counts[f"n/img_{batch['kind']}"] += size


@torch.no_grad()
def _label_ranking_metrics(
    z: torch.Tensor,
    labels: list[str],
    candidate_labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    per_class_out: list[dict[str, object]] | None = None,
    world_size: int = 1,
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
    recall_at_5_count = torch.bincount(
        true,
        weights=recall_at_5.float(),
        minlength=num_classes,
    )
    packed = torch.cat(
        (
            support.double(),
            predicted.double(),
            true_positive.double(),
            recall_at_5_count.double(),
            reciprocal_rank.double().sum().reshape(1),
        )
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    support, predicted, true_positive, recall_at_5_count = packed[:-1].reshape(4, num_classes)
    sample_count = support.sum()

    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    recall_at_5_by_class = recall_at_5_count / support.clamp_min(1)
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
        "accuracy": float(true_positive.sum() / sample_count),
        "f1_macro": float(f1[supported].mean()),
        "recall_at_1": float(true_positive.sum() / sample_count),
        "recall_at_5": float(recall_at_5_count.sum() / sample_count),
        "map": float(packed[-1] / sample_count),
        "prediction_coverage": float((predicted > 0).sum()) / num_classes,
    }


@torch.no_grad()
def _ts_prediction_metrics(
    z: torch.Tensor,
    texts: list[str],
    families: list[str],
    label_bank: dict[str, tuple[tuple[str, ...], torch.Tensor]],
    per_class_reports: dict[str, list[dict[str, object]]] | None = None,
    world_size: int = 1,
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
            world_size=world_size,
            per_class_out=report_rows,
        )
        family_scores[family] = family_metrics
        for stat in (*_PUBLIC_STATS, "prediction_coverage"):
            metrics[f"{stat}/ts_{family}"] = family_metrics[stat]

    for stat in _PUBLIC_STATS:
        values = [scores[stat] for scores in family_scores.values() if stat in scores]
        if values:
            metrics[f"{stat}/overall"] = sum(values) / len(values)

    return metrics


@torch.no_grad()
def _tactile_task_metrics(
    task: str,
    z: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    bias: float,
    log_logit_scale: torch.Tensor,
    world_size: int,
    per_label_out: list[dict[str, object]] | None,
) -> dict[str, float]:
    """Score one structured tactile task, including multi-positive labels."""
    num_labels = len(labels)
    mask = mask.to(z.device)
    target = targets.to(z.device)[mask].bool()
    similarities = z[mask].float() @ text_embeddings.to(z.device).float().t()
    logits = log_logit_scale.float().exp() * similarities + bias
    ranked = similarities.argsort(dim=1, descending=True)
    top = ranked[:, 0]
    top_is_positive = target.gather(1, top[:, None]).squeeze(1)
    top_five_is_positive = target.gather(
        1,
        ranked[:, : min(5, num_labels)],
    ).any(dim=1)

    ranked_target = target.gather(1, ranked).float()
    ranks = torch.arange(1, num_labels + 1, device=z.device, dtype=torch.float32)
    precision_at_rank = ranked_target.cumsum(dim=1) / ranks
    average_precision = (precision_at_rank * ranked_target).sum(dim=1) / (ranked_target.sum(dim=1).clamp_min(1))

    if task in MULTILABEL_TASKS:
        predicted_mask = logits > 0
    else:
        predicted_mask = F.one_hot(top, num_classes=num_labels).bool()
    exact_match = (predicted_mask == target).all(dim=1)
    support = target.sum(dim=0).double()
    predicted = predicted_mask.sum(dim=0).double()
    true_positive = (target & predicted_mask).sum(dim=0).double()
    packed = torch.cat(
        (
            support,
            predicted,
            true_positive,
            exact_match.double().sum().reshape(1),
            top_is_positive.double().sum().reshape(1),
            top_five_is_positive.double().sum().reshape(1),
            average_precision.double().sum().reshape(1),
            target.new_tensor([len(target)], dtype=torch.float64),
        )
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    support, predicted, true_positive = packed[:-5].reshape(3, num_labels)
    sample_count = packed[-1]
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0

    if per_label_out is not None:
        per_label_out.extend(
            {
                "task": task,
                "label": label,
                "support": int(support[index].item()),
                "predicted": int(predicted[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
            for index, label in enumerate(labels)
        )

    return {
        "accuracy": float(packed[-5] / sample_count),
        "f1_macro": float(f1[supported].mean()),
        "recall_at_1": float(packed[-4] / sample_count),
        "recall_at_5": float(packed[-3] / sample_count),
        "map": float(packed[-2] / sample_count),
        "prediction_coverage": float((predicted > 0).sum()) / num_labels,
    }


@torch.no_grad()
def _tactile_prediction_metrics(
    z: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    label_bank: dict[str, tuple[tuple[str, ...], torch.Tensor, float]],
    log_logit_scale: torch.Tensor,
    *,
    world_size: int = 1,
    per_label_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Compute equal-task structured metrics for the tactile family."""
    metrics: dict[str, float] = {}
    task_scores: dict[str, dict[str, float]] = {}
    for task, (labels, text_embeddings, bias) in label_bank.items():
        scores = _tactile_task_metrics(
            task,
            z,
            targets[task],
            masks[task],
            labels,
            text_embeddings,
            bias,
            log_logit_scale,
            world_size,
            per_label_out,
        )
        task_scores[task] = scores
        for name, value in scores.items():
            metrics[f"{name}/task/{task}"] = value

    for stat in (*_PUBLIC_STATS, "prediction_coverage"):
        metrics[f"{stat}/ts_tactile"] = sum(scores[stat] for scores in task_scores.values()) / len(task_scores)
    return metrics


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
    prediction_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build the shared train/validation W&B metric surface."""
    prediction_metrics = loss_metrics if prediction_metrics is None else prediction_metrics
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
        for stat in (*_PUBLIC_STATS, "prediction_coverage"):
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
