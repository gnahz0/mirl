"""Sensor-to-text ranking metrics and their single distributed reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import (
    _CLASSIFICATION_FAMILIES,
    _TS_FAMILIES,
    MULTILABEL_TASKS,
    TACTILE_SPANS,
    TASK_LABELS,
)

_PUBLIC_STATS = ("accuracy", "recall_at_1", "recall_at_5", "map")
_AUX_STATS = ("prediction_coverage",)
_PREDICTION_STATS = _PUBLIC_STATS + _AUX_STATS

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
)


def all_reduce_sum(values: dict[str, torch.Tensor], *, world_size: int = 1) -> dict[str, torch.Tensor]:
    """Sum every entry of ``values`` across ranks with exactly one collective.
    Callers fix the key set before any data is seen, so the packed buffer is
    identical on every rank whatever rows landed locally."""
    if world_size <= 1:
        return values
    keys = sorted(values)
    flat = torch.cat([values[key].reshape(-1).double() for key in keys])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    return {
        key: chunk.view(values[key].shape)
        for key, chunk in zip(keys, flat.split([values[key].numel() for key in keys]))
    }


@dataclass(frozen=True)
class BankSpec:
    """One scoring unit: some rows ranked against some bank of label embeddings."""

    key: str  # "ts_smellnet" | "task/force_level"
    labels: tuple[str, ...]
    embeddings: torch.Tensor  # (K, D) frozen bank or bank slice
    multilabel: bool = False


def build_bank_specs(label_bank: dict, tactile_bank: tuple | None) -> tuple[BankSpec, ...]:
    """Enumerate every scoring unit once, before any data is seen, so the unit list
    -- and the reduce buffer's length -- is identical on every rank for the run.
    ``update_stats`` re-reads ``TACTILE_SPANS``; both must see the same span table."""
    specs = [BankSpec(f"ts_{family}", *label_bank[family]) for family in _TS_FAMILIES if family in label_bank]
    if tactile_bank is not None:
        labels, embeddings = tactile_bank
        specs += [
            BankSpec(f"task/{task}", labels[start:stop], embeddings[start:stop], task in MULTILABEL_TASKS)
            for task, (start, stop) in TACTILE_SPANS.items()
        ]
    return tuple(specs)


@torch.no_grad()
def _bank_stats(
    z: torch.Tensor,
    target: torch.Tensor,
    spec: BankSpec,
    log_logit_scale: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sufficient statistics for one (rows x bank) scoring problem. No collective.
    Sums add across microbatches exactly as across ranks, so nothing retains
    embeddings; zero rows legally yields zeros (``new_stats`` sizes the buffer with it)."""
    target = target.to(z.device).bool()
    # A transposed view, never .contiguous(): fresh strides change the cuBLAS kernel and can flip tie order.
    sims = z.float() @ spec.embeddings.to(z.device).float().t()
    # stable=True pins tie order to label index, so a row scores identically however the rows were chunked.
    ranked = sims.argsort(dim=1, descending=True, stable=True)
    top, top_k = ranked[:, :1], ranked[:, :5]  # a slice already clamps to a narrower bank

    top_is_positive = target.gather(1, top).squeeze(1)
    positive_count = target.sum(dim=1).clamp_min(1)
    recall_at_1 = top_is_positive.float() / positive_count
    recall_at_5 = target.gather(1, top_k).sum(dim=1) / positive_count

    ranked_target = target.gather(1, ranked).float()
    ranks = torch.arange(1, len(spec.labels) + 1, device=z.device, dtype=torch.float32)
    precision_at_rank = ranked_target.cumsum(dim=1) / ranks
    average_precision = (precision_at_rank * ranked_target).sum(dim=1) / ranked_target.sum(dim=1).clamp_min(1)

    predicted = (
        log_logit_scale.float().exp() * sims + logit_bias.float() > 0
        if spec.multilabel
        else torch.zeros_like(target).scatter_(1, top, True)
    )
    in_top_k = torch.zeros_like(target).scatter_(1, top_k, True)

    return {
        f"{spec.key}/support": target.sum(dim=0).double(),
        f"{spec.key}/predicted": predicted.sum(dim=0).double(),
        f"{spec.key}/true_positive": (target & predicted).sum(dim=0).double(),
        f"{spec.key}/recall_at_5_count": (target & in_top_k).sum(dim=0).double(),
        f"{spec.key}/top_is_positive": top_is_positive.double().sum(),
        f"{spec.key}/recall_at_1": recall_at_1.double().sum(),
        f"{spec.key}/recall_at_5": recall_at_5.double().sum(),
        f"{spec.key}/average_precision": average_precision.double().sum(),
        f"{spec.key}/n": z.new_tensor(float(len(target)), dtype=torch.float64),
    }


@torch.no_grad()
def _bank_metrics(
    stats: dict[str, torch.Tensor],
    spec: BankSpec,
    *,
    rows_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Turn globally reduced sufficient statistics into the published scalars."""
    support, predicted = stats[f"{spec.key}/support"], stats[f"{spec.key}/predicted"]
    true_positive, sample_count = stats[f"{spec.key}/true_positive"], stats[f"{spec.key}/n"]

    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    recall_at_5_by_class = stats[f"{spec.key}/recall_at_5_count"] / support.clamp_min(1)
    task = spec.key.partition("task/")[2]

    if rows_out is not None:
        rows_out.extend(
            {
                # The six tasks share one table, so their rows say which task; a family table would only repeat its name.
                **({"task": task} if task else {}),
                "class_id": index,
                "label": label,
                "support": support[index].item(),
                "predicted": predicted[index].item(),
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "recall_at_5": recall_at_5_by_class[index].item(),
            }
            for index, label in enumerate(spec.labels)
        )

    return {
        "accuracy": float(stats[f"{spec.key}/top_is_positive"] / sample_count),
        "recall_at_1": float(stats[f"{spec.key}/recall_at_1"] / sample_count),
        "recall_at_5": float(stats[f"{spec.key}/recall_at_5"] / sample_count),
        "map": float(stats[f"{spec.key}/average_precision"] / sample_count),
        "prediction_coverage": float((predicted > 0).sum() / len(spec.labels)),
    }


def new_stats(specs: tuple[BankSpec, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Zero-fill the complete statistic key set before any data is seen, sized by
    the producer's zero-row statistics so the buffer cannot fall out of step with
    what ``update_stats`` writes."""
    return {
        key: value.to(device)
        for spec in specs
        for key, value in _bank_stats(
            spec.embeddings.new_zeros(0, spec.embeddings.shape[1]),
            spec.embeddings.new_zeros(0, len(spec.labels)),
            spec,
            spec.embeddings.new_zeros(()),
            spec.embeddings.new_zeros(()),
        ).items()
    }


def _microbatch_units(specs, z, texts, families, targets, masks):
    """Yield (spec, rows of ``z``, multi-hot target) for every unit with rows here.
    The one place that tells family label strings from tactile answered spans apart;
    a visual microbatch carries no families, so it yields nothing at all."""
    by_key = {spec.key: spec for spec in specs}
    if masks is None:
        rows_by_key: dict[str, list[int]] = {}
        for position, family in enumerate(families):
            rows_by_key.setdefault(f"ts_{family}", []).append(position)
        for key, rows in rows_by_key.items():
            spec = by_key[key]
            ids = torch.tensor([spec.labels.index(texts[row]) for row in rows], dtype=torch.long)
            yield spec, z[rows], F.one_hot(ids, len(spec.labels))
        return
    for task, (start, stop) in TACTILE_SPANS.items():
        # collate writes whole task spans; requiring the whole span drops a partly-answered row, not misreads it.
        rows = (masks[:, start:stop] > 0).all(dim=1)
        yield by_key[f"task/{task}"], z[rows], targets[rows, start:stop]


@torch.no_grad()
def update_stats(
    stats: dict[str, torch.Tensor],
    specs: tuple[BankSpec, ...],
    ts_eval: tuple,
    log_logit_scale: torch.Tensor | None,
    logit_bias: torch.Tensor | None,
) -> None:
    """Fold one microbatch into the running statistics. No collective, no retention."""
    for spec, rows, target in _microbatch_units(specs, *ts_eval):
        for key, value in _bank_stats(rows, target, spec, log_logit_scale, logit_bias).items():
            # Fixed key set: a diverged rank raises KeyError rather than hanging.
            stats[key] += value


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
        if not float(stats[f"{spec.key}/n"]):
            continue
        is_task = spec.key.startswith("task/")
        # The six tasks share one table; each reported family gets its own.
        rows_out = per_label if is_task else None
        if per_class is not None and (family := spec.key.removeprefix("ts_")) in _CLASSIFICATION_FAMILIES:
            rows_out = per_class.setdefault(family, [])
        scores = _bank_metrics(stats, spec, rows_out=rows_out)
        for stat in _PREDICTION_STATS:
            metrics[f"{stat}/{spec.key}"] = scores[stat]
        if is_task:
            task_scores.append(scores)
    if task_scores:
        for stat in _PREDICTION_STATS:
            metrics[f"{stat}/ts_tactile"] = sum(s[stat] for s in task_scores) / len(task_scores)
    return _merge_prediction_metrics(metrics)


def _merge_prediction_metrics(*metric_sets: dict[str, float]) -> dict[str, float]:
    """Merge family metrics and recompute an equal-family overall score."""
    merged: dict[str, float] = {}
    for values in metric_sets:
        merged.update(values)
    for stat in _PREDICTION_STATS:
        family_values = [merged[f"{stat}/ts_{family}"] for family in _TS_FAMILIES if f"{stat}/ts_{family}" in merged]
        if family_values:
            merged[f"{stat}/overall"] = sum(family_values) / len(family_values)
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
        for stat in _AUX_STATS:
            key = f"{stat}/ts_{family}"
            if key in prediction_metrics:
                out[f"{aux}/{stat}/{family}"] = prediction_metrics[key]
        out[f"{aux}/n/{family}"] = float(counts[f"n/ts_{family}"])

    for task in TASK_LABELS:
        loss_key = f"loss/task/{task}"
        if loss_key in loss_metrics:
            out[f"{aux}/loss/tactile/{task}"] = loss_metrics[loss_key]
        for stat in _PREDICTION_STATS:
            key = f"{stat}/task/{task}"
            if key in prediction_metrics:
                out[f"{aux}/{stat}/tactile/{task}"] = prediction_metrics[key]

    for stat in _PUBLIC_STATS:
        key = f"{stat}/overall"
        if key in prediction_metrics:
            out[f"{core}/{stat}/overall"] = prediction_metrics[key]
    for stat in _AUX_STATS:
        key = f"{stat}/overall"
        if key in prediction_metrics:
            out[f"{aux}/{stat}/overall"] = prediction_metrics[key]

    if split == "train":
        for key, value in loss_metrics.items():
            if key in {"grad_norm", "logit_scale", "logit_bias"}:
                out[f"{aux}/{key}"] = value
        out[f"{aux}/n/img"] = float(counts["n/img_image"] + counts["n/img_video"])

    return out
