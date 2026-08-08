# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Structured tactile metrics and distributed loss reduction."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import MULTILABEL_TASKS, TASK_LABELS

_PUBLIC_STATS = ("f1_macro", "recall_at_1", "map")
_REDUCED_METRIC_KEYS = (
    "loss/siglip",
    "loss/total",
    *(f"loss/task/{task}" for task in TASK_LABELS),
)
_COUNT_KEYS = ("n/tactile",)


def _allreduce_metrics(metrics: dict, device: torch.device, world_size: int) -> dict:
    missing = [key for key in metrics if key.startswith("loss/") and key not in _REDUCED_METRIC_KEYS]
    if missing:
        raise RuntimeError(f"metrics absent from _REDUCED_METRIC_KEYS: {missing}")
    if world_size <= 1:
        return metrics

    flat: list[float] = []
    for key in _REDUCED_METRIC_KEYS:
        value = metrics.get(key)
        flat += [float(value), 1.0] if value is not None else [0.0, 0.0]
    packed = torch.tensor(flat, device=device, dtype=torch.float64)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.tolist()
    out = dict(metrics)
    for index, key in enumerate(_REDUCED_METRIC_KEYS):
        total, present = reduced[2 * index], reduced[2 * index + 1]
        if present:
            out[key] = total / present
        else:
            out.pop(key, None)
    return out


def _allreduce_counts(counts: dict, device: torch.device, world_size: int) -> dict:
    if world_size <= 1:
        return {key: int(counts.get(key, 0)) for key in _COUNT_KEYS}
    packed = torch.tensor(
        [counts.get(key, 0) for key in _COUNT_KEYS],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {key: int(value) for key, value in zip(_COUNT_KEYS, packed.tolist(), strict=True)}


def add_batch_counts(counts: dict, batch: dict) -> None:
    counts["n/tactile"] += len(batch["media"])


@torch.no_grad()
def _one_task_metrics(
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
    num_labels = len(labels)
    target = targets[mask].bool()
    similarities = z[mask].float() @ text_embeddings.to(z.device).float().t()
    logits = log_logit_scale.float().exp() * similarities + bias
    ranked = similarities.argsort(dim=1, descending=True)
    top = ranked[:, 0]
    top_is_positive = target.gather(1, top[:, None]).squeeze(1)

    ranked_target = target.gather(1, ranked).float()
    rank = torch.arange(1, num_labels + 1, device=z.device, dtype=torch.float32)
    precision_at_rank = ranked_target.cumsum(dim=1) / rank
    average_precision = (precision_at_rank * ranked_target).sum(dim=1) / ranked_target.sum(dim=1).clamp_min(1)

    if task in MULTILABEL_TASKS:
        predicted_mask = logits > 0
    else:
        predicted_mask = F.one_hot(top, num_classes=num_labels).bool()
    support = target.sum(dim=0).double()
    predicted = predicted_mask.sum(dim=0).double()
    true_positive = (target & predicted_mask).sum(dim=0).double()
    packed = torch.cat(
        (
            support,
            predicted,
            true_positive,
            top_is_positive.double().sum().reshape(1),
            average_precision.double().sum().reshape(1),
            target.new_tensor([len(target)], dtype=torch.float64),
        )
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    support, predicted, true_positive = packed[:-3].reshape(3, num_labels)
    sample_count = packed[-1]
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0

    if per_label_out is not None:
        per_label_out.extend(
            {
                "task": task,
                "label": label,
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(labels)
        )

    metrics = {
        "f1_macro": float(f1[supported].mean()),
        "recall_at_1": float(packed[-3] / sample_count),
        "map": float(packed[-2] / sample_count),
    }
    if task not in MULTILABEL_TASKS:
        metrics["accuracy"] = metrics["recall_at_1"]
    return metrics


@torch.no_grad()
def task_prediction_metrics(
    z: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    label_bank: dict[str, tuple[tuple[str, ...], torch.Tensor, float]],
    log_logit_scale: torch.Tensor,
    *,
    world_size: int = 1,
    per_label_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    task_scores: dict[str, dict[str, float]] = {}
    for task, (labels, text_embeddings, bias) in label_bank.items():
        scores = _one_task_metrics(
            task,
            z,
            targets[task].to(z.device),
            masks[task].to(z.device),
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

    for stat in _PUBLIC_STATS:
        metrics[f"{stat}/tactile"] = sum(scores[stat] for scores in task_scores.values()) / len(task_scores)
    return metrics


def metric_groups(
    split: str,
    loss_metrics: dict[str, float],
    counts: dict[str, int],
    prediction_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    prediction_metrics = loss_metrics if prediction_metrics is None else prediction_metrics
    core = f"{split}-core"
    aux = f"{split}-aux"
    out = {
        f"{core}/loss/aggregate": loss_metrics["loss/total"],
        f"{aux}/n/tactile": float(counts["n/tactile"]),
    }
    if "loss/siglip" in loss_metrics:
        out[f"{aux}/loss/siglip"] = loss_metrics["loss/siglip"]

    for task in TASK_LABELS:
        loss_key = f"loss/task/{task}"
        if loss_key in loss_metrics:
            out[f"{aux}/loss/{task}"] = loss_metrics[loss_key]
        for stat in (*_PUBLIC_STATS, "accuracy"):
            key = f"{stat}/task/{task}"
            if key in prediction_metrics:
                out[f"{aux}/{stat}/{task}"] = prediction_metrics[key]

    for stat in _PUBLIC_STATS:
        key = f"{stat}/tactile"
        if key in prediction_metrics:
            out[f"{core}/{stat}/tactile"] = prediction_metrics[key]

    if split == "train":
        for key, value in loss_metrics.items():
            if key.startswith("grad_norm") or key == "logit_scale":
                out[f"{aux}/{key}"] = value
    return out
