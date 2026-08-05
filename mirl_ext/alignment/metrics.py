# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Time-series prototype metrics and distributed metric reduction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

# Metric reduction order must be identical on every rank.
_TS_FAMILIES: tuple[str, ...] = ("smell", "ecg", "tactile")
_METRIC_FAMILY_NAMES = {
    "smell": "smellnet",
    "ecg": "ecg",
    "tactile": "haptic",
}

_REDUCED_METRIC_KEYS: tuple[str, ...] = (
    (
        "loss/ts_text",
        "loss/ts_smell",
        "loss/ts_ecg",
        "loss/ts_tactile",
        "loss/smell_sensor_gcms",
        "loss/smell_gcms_text",
        "loss/distill_img",
        "loss/total",
        # Equal-family core metrics only include families with a real prototype task.
        *(
            f"{stat}/ts_supervised_family_macro"
            for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        ),
        *(
            f"{stat}/ts_supervised_class_macro"
            for stat in ("precision_macro", "recall_macro", "f1_macro")
        ),
        *(
            f"{stat}/ts_supervised_{aggregation}_macro"
            for aggregation in ("family", "class")
            for stat in ("recall_at_1_macro", "recall_at_5_macro")
        ),
        "accuracy/smell_sensor_to_gcms",
        "accuracy/smell_gcms_to_text",
    )
    + tuple(
        f"{stat}/ts_{family}"
        for family in _TS_FAMILIES
        for stat in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "recall_at_1",
            "recall_at_5",
            "recall_at_1_macro",
            "recall_at_5_macro",
            "gap",
            "eff_dim",
            "label_coverage",
            "class_coverage",
            "prediction_coverage",
        )
    )
)

_COUNT_KEYS: tuple[str, ...] = (
    "n/img_image",
    "n/img_video",
    "n/ts_signal",
) + tuple(f"n/ts_{family}" for family in _TS_FAMILIES)

_MUST_REDUCE_PREFIXES = (
    "loss/",
    "accuracy/",
    "precision_macro/",
    "recall_macro/",
    "f1_macro/",
    "recall_at_",
    "gap/",
    "eff_dim/",
    "coverage/",
    "label_coverage/",
    "class_coverage/",
    "prediction_coverage/",
)

# Recompute nonlinear metrics over the complete accumulation window.
_TS_WINDOW_METRIC_PREFIXES = (
    "accuracy/",
    "precision_macro/",
    "recall_macro/",
    "f1_macro/",
    "recall_at_",
    "map/",
    "gap/",
    "eff_dim/",
    "label_coverage/",
    "class_coverage/",
    "prediction_coverage/",
)


def new_counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _is_ts_window_metric(key: str) -> bool:
    return key.startswith(_TS_WINDOW_METRIC_PREFIXES)


def _allreduce_metrics(metrics: dict, device: torch.device, world_size: int) -> dict:
    """Average each present metric over the ranks that computed it."""
    missing = [key for key in metrics if key.startswith(_MUST_REDUCE_PREFIXES) and key not in _REDUCED_METRIC_KEYS]
    if missing:
        raise RuntimeError(f"metrics absent from _REDUCED_METRIC_KEYS: {missing}")
    if world_size <= 1:
        return metrics
    # Build host-side then transfer once: assigning element-by-element into a CUDA
    # tensor costs one H2D write + kernel launch each, and reading back with
    # per-element .item() costs one device sync each (~27 round-trips for 120 bytes).
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


def group_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return L2 gradient norms for the ViT, heads, and scalar tiers."""
    tiers: dict[str, list[torch.Tensor]] = {"vit": [], "head": [], "scalar": []}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if name.startswith("trainable_visual."):
            tier = "vit"
        elif param.ndim == 0:
            tier = "scalar"
        else:
            tier = "head"
        tiers[tier].append(param.grad.detach())
    # `_foreach_norm` is one multi-tensor kernel per tier -- the same primitive
    # `clip_grad_norm_` uses. A per-parameter `float(...)` would sync the device once
    # per tensor, i.e. ~450 syncs every step on a 456M tower.
    out = {}
    for tier, grads in tiers.items():
        if not grads:
            out[f"grad_norm/{tier}"] = 0.0
            continue
        out[f"grad_norm/{tier}"] = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))))
    return out


@torch.no_grad()
def _smellnet_gcms_top1_metrics(
    z_ts: torch.Tensor,
    labels: list[str],
    families: list[str],
    text_bank: tuple[tuple[str, ...], torch.Tensor],
    gcms_bank: tuple[tuple[str, ...], torch.Tensor],
) -> dict[str, float]:
    """Report sensor-to-GC-MS and GC-MS-to-text top-1 retrieval accuracy."""
    text_labels, text_features = text_bank
    gcms_labels, gcms_features = gcms_bank

    out: dict[str, float] = {}
    smell_rows = [index for index, family in enumerate(families) if family == "smell"]
    if smell_rows:
        select = torch.tensor(smell_rows, device=z_ts.device, dtype=torch.long)
        smell_labels = [labels[index] for index in smell_rows]
        result = _prototype_classification_metrics(
            z_ts.index_select(0, select),
            smell_labels,
            gcms_labels,
            gcms_features,
        )
        if "accuracy" in result:
            out["accuracy/smell_sensor_to_gcms"] = result["accuracy"]

    similarity = gcms_features.float() @ text_features.float().t()
    target = torch.arange(len(gcms_labels), device=similarity.device)
    out["accuracy/smell_gcms_to_text"] = float((similarity.argmax(dim=1) == target).float().mean())
    return out


def add_ts_family_counts(counts: dict, batch: dict) -> None:
    """Add per-family row counts."""
    for family in batch.get("ts_format") or []:
        counts[f"n/ts_{family}"] += 1


@torch.no_grad()
def _effective_dim(z: torch.Tensor) -> Optional[float]:
    """Participation ratio of the centered embedding covariance spectrum."""
    if z is None or z.shape[0] < 3:
        return None
    centered = z.float() - z.float().mean(dim=0, keepdim=True)
    ev = torch.linalg.svdvals(centered) ** 2
    total = float(ev.sum())
    if total <= 0.0:
        return None
    return float((total**2) / float((ev**2).sum()))


@torch.no_grad()
def _paired_retrieval_metrics(
    sensor: torch.Tensor,
    text: torch.Tensor,
    captions: list[str],
    family: str,
) -> dict[str, float]:
    """Score both directions of paired sensor-text retrieval."""
    similarity = sensor.float() @ text.float().t()
    positive = torch.tensor(
        [[left == right for right in captions] for left in captions],
        device=similarity.device,
        dtype=torch.bool,
    )
    metrics: dict[str, float] = {}
    for direction, scores, matches in (
        ("sensor_to_text", similarity, positive),
        ("text_to_sensor", similarity.t(), positive.t()),
    ):
        ranked = matches.gather(1, scores.argsort(dim=1, descending=True))
        for k in (1, 5):
            recall = ranked[:, : min(k, ranked.shape[1])].any(dim=1).float().mean()
            metrics[f"recall_at_{k}/ts_{family}_{direction}"] = float(recall)

        ranks = torch.arange(1, ranked.shape[1] + 1, device=ranked.device)
        precision = ranked.cumsum(dim=1) / ranks
        average_precision = (precision * ranked).sum(dim=1) / ranked.sum(dim=1)
        metrics[f"map/ts_{family}_{direction}"] = float(average_precision.mean())
    return metrics


@torch.no_grad()
def _prototype_classification_metrics(
    z: torch.Tensor,
    labels: list[str],
    prototype_labels: tuple[str, ...],
    prototypes: torch.Tensor,
    per_class_out: Optional[list[dict[str, object]]] = None,
) -> dict[str, float]:
    """Score known labels, macro-averaging only classes present in ground truth."""
    if not labels or not prototype_labels:
        return {
            "label_coverage": 0.0,
            "class_coverage": 0.0,
            "prediction_coverage": 0.0,
        }

    label_to_id = {label: idx for idx, label in enumerate(prototype_labels)}
    prototypes = prototypes.to(device=z.device)
    true = torch.tensor(
        [label_to_id.get(label, -1) for label in labels],
        device=z.device,
        dtype=torch.long,
    )
    known = true >= 0
    result = {
        "label_coverage": float(known.float().mean()),
        "class_coverage": 0.0,
        "prediction_coverage": 0.0,
    }
    if not bool(known.any()):
        if per_class_out is not None:
            per_class_out.extend(
                {
                    "class_id": class_id,
                    "label": label,
                    "support": 0,
                    "predicted": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "recall_at_5": 0.0,
                }
                for class_id, label in enumerate(prototype_labels)
            )
        return result

    known_z = z[known].float()
    known_true = true[known]
    sims = known_z @ prototypes.float().t()
    ranked = sims.argsort(dim=1, descending=True)
    pred = ranked[:, 0]
    recall_at_5 = (ranked[:, : min(5, ranked.shape[1])] == known_true[:, None]).any(dim=1)
    num_classes = len(prototype_labels)
    support = torch.bincount(known_true, minlength=num_classes)
    predicted = torch.bincount(pred, minlength=num_classes)
    true_positive = torch.bincount(
        known_true[pred == known_true],
        minlength=num_classes,
    )

    per_class: list[dict[str, object]] = []
    supported_precision: list[float] = []
    supported_recall: list[float] = []
    supported_f1: list[float] = []
    supported_recall_at_5: list[float] = []
    for class_id, label in enumerate(prototype_labels):
        tp = int(true_positive[class_id])
        support_i = int(support[class_id])
        predicted_i = int(predicted[class_id])
        precision = tp / predicted_i if predicted_i else 0.0
        recall = tp / support_i if support_i else 0.0
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom else 0.0
        class_mask = known_true == class_id
        class_recall_at_5 = float(recall_at_5[class_mask].float().mean()) if support_i else 0.0
        if support_i:
            supported_precision.append(precision)
            supported_recall.append(recall)
            supported_f1.append(f1)
            supported_recall_at_5.append(class_recall_at_5)
        per_class.append(
            {
                "class_id": class_id,
                "label": label,
                "support": support_i,
                "predicted": predicted_i,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "recall_at_5": class_recall_at_5,
            }
        )
    if per_class_out is not None:
        per_class_out.extend(per_class)

    result.update(
        {
            "accuracy": float((pred == known_true).float().mean()),
            "precision_macro": sum(supported_precision) / len(supported_precision),
            "recall_macro": sum(supported_recall) / len(supported_recall),
            "f1_macro": sum(supported_f1) / len(supported_f1),
            "recall_at_1": float((pred == known_true).float().mean()),
            "recall_at_5": float(recall_at_5.float().mean()),
            "recall_at_1_macro": sum(supported_recall) / len(supported_recall),
            "recall_at_5_macro": sum(supported_recall_at_5) / len(supported_recall_at_5),
            "class_coverage": len(supported_f1) / num_classes,
            "prediction_coverage": float((predicted > 0).sum()) / num_classes,
            "supported_classes": len(supported_f1),
        }
    )

    pos = sims.gather(1, known_true[:, None]).squeeze(1)
    negative_mask = torch.ones_like(sims, dtype=torch.bool)
    negative_mask.scatter_(1, known_true[:, None], False)
    neg = sims[negative_mask]
    result["pos_sim"] = float(pos.mean())
    result["neg_sim"] = float(neg.mean()) if neg.numel() else 0.0
    result["gap"] = result["pos_sim"] - result["neg_sim"]
    return result


@torch.no_grad()
def _ts_prediction_metrics(
    z: torch.Tensor,
    texts: list[str],
    families: list[str],
    prototype_bank: dict[str, tuple[tuple[str, ...], torch.Tensor]],
    per_class_reports: Optional[dict[str, list[dict[str, object]]]] = None,
) -> dict[str, float]:
    """Compute honest per-family metrics and an equal-family overall score."""
    metrics: dict[str, float] = {}
    family_cms: dict[str, dict] = {}
    for family in _TS_FAMILIES:
        idx = [i for i, value in enumerate(families) if value == family]
        if not idx:
            continue
        sel = torch.tensor(idx, device=z.device, dtype=torch.long)
        family_labels = [texts[i] for i in idx]
        entry = prototype_bank.get(family)
        if entry is not None:
            report_rows = per_class_reports.setdefault(family, []) if per_class_reports is not None else None
            fam_cm = _prototype_classification_metrics(
                z[sel],
                family_labels,
                entry[0],
                entry[1],
                per_class_out=report_rows,
            )
            family_cms[family] = fam_cm
            for stat in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "recall_at_1",
                "recall_at_5",
                "recall_at_1_macro",
                "recall_at_5_macro",
                "gap",
                "label_coverage",
                "class_coverage",
                "prediction_coverage",
            ):
                if stat in fam_cm:
                    metrics[f"{stat}/ts_{family}"] = fam_cm[stat]
        fam_eff = _effective_dim(z[sel])
        if fam_eff is not None:
            metrics[f"eff_dim/ts_{family}"] = fam_eff

    for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        available = [family for family in _TS_FAMILIES if stat in family_cms.get(family, {})]
        if available:
            metrics[f"{stat}/ts_supervised_family_macro"] = sum(family_cms[family][stat] for family in available) / len(
                available
            )

    for stat in ("precision_macro", "recall_macro", "f1_macro"):
        available = [family for family in _TS_FAMILIES if stat in family_cms.get(family, {})]
        total_classes = sum(family_cms[family]["supported_classes"] for family in available)
        if total_classes:
            metrics[f"{stat}/ts_supervised_class_macro"] = sum(
                family_cms[family][stat] * family_cms[family]["supported_classes"]
                for family in available
            ) / total_classes

    for stat in ("recall_at_1_macro", "recall_at_5_macro"):
        available = [family for family in _TS_FAMILIES if stat in family_cms.get(family, {})]
        if available:
            metrics[f"{stat}/ts_supervised_family_macro"] = sum(
                family_cms[family][stat] for family in available
            ) / len(available)
        total_classes = sum(family_cms[family]["supported_classes"] for family in available)
        if total_classes:
            metrics[f"{stat}/ts_supervised_class_macro"] = sum(
                family_cms[family][stat] * family_cms[family]["supported_classes"]
                for family in available
            ) / total_classes

    return metrics


def _training_metric_groups(
    metrics: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    """Return the compact, public training metric surface for W&B."""
    out: dict[str, float] = {"train/loss": metrics["loss/total"]}
    for source, display in (("loss/ts_text", "siglip"), ("loss/distill_img", "distill")):
        if source in metrics:
            out[f"train-aux/loss/{display}"] = metrics[source]
    for family in _TS_FAMILIES:
        display = _METRIC_FAMILY_NAMES[family]
        loss_key = f"loss/ts_{family}"
        if loss_key in metrics:
            out[f"train-aux/loss/siglip/{display}"] = metrics[loss_key]
        for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            key = f"{stat}/ts_{family}"
            if key in metrics:
                out[f"train-core/{stat}/{display}"] = metrics[key]
        for stat in ("recall_at_1", "recall_at_5", "map"):
            for direction in ("sensor_to_text", "text_to_sensor"):
                key = f"{stat}/ts_{family}_{direction}"
                if key in metrics:
                    out[f"train-core/{stat}/{display}_{direction}"] = metrics[key]
        for stat in ("recall_at_1", "recall_at_5", "recall_at_1_macro", "recall_at_5_macro"):
            key = f"{stat}/ts_{family}"
            if key in metrics:
                out[f"train-core/{stat}/{display}"] = metrics[key]
        for source, target in (
            ("eff_dim", "effective_dimension"),
            ("prediction_coverage", "prediction_coverage"),
        ):
            key = f"{source}/ts_{family}"
            if key in metrics:
                out[f"train-aux/{target}/{display}"] = metrics[key]
        out[f"train-aux/n/{display}"] = float(counts[f"n/ts_{family}"])

    for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        key = f"{stat}/ts_supervised_family_macro"
        if key in metrics:
            out[f"train-core/{stat}/overall"] = metrics[key]
    for stat in ("precision_macro", "recall_macro", "f1_macro"):
        key = f"{stat}/ts_supervised_class_macro"
        if key in metrics:
            out[f"train-core/{stat}/all_classes"] = metrics[key]
    for stat in ("recall_at_1_macro", "recall_at_5_macro"):
        family_key = f"{stat}/ts_supervised_family_macro"
        class_key = f"{stat}/ts_supervised_class_macro"
        if family_key in metrics:
            out[f"train-core/{stat}/overall"] = metrics[family_key]
        if class_key in metrics:
            out[f"train-core/{stat}/all_classes"] = metrics[class_key]

    for key, value in metrics.items():
        if key.startswith("grad_norm/") or key in {"grad_norm", "logit_scale"}:
            out[f"train-aux/{key}"] = value
    for source, target in (
        ("accuracy/smell_sensor_to_gcms", "smellnet_sensor_to_gcms"),
        ("accuracy/smell_gcms_to_text", "smellnet_gcms_to_text"),
    ):
        if source in metrics:
            out[f"train-aux/accuracy/{target}"] = metrics[source]
    out["train-aux/n/img"] = float(counts["n/img_image"] + counts["n/img_video"])
    return out


def _validation_metric_groups(
    averaged_metrics: dict[str, float],
    prediction_metrics: dict[str, float],
    bucket_totals: dict[str, int],
) -> dict[str, float]:
    """Return core selection metrics and a small set of validation diagnostics."""
    out = {"val/loss": averaged_metrics["loss/total"]}
    for source, display in (("loss/ts_text", "siglip"), ("loss/distill_img", "distill")):
        if source in averaged_metrics:
            out[f"val-aux/loss/{display}"] = averaged_metrics[source]
    for family in _TS_FAMILIES:
        display = _METRIC_FAMILY_NAMES[family]
        loss_key = f"loss/ts_{family}"
        if loss_key in averaged_metrics:
            out[f"val-aux/loss/siglip/{display}"] = averaged_metrics[loss_key]
        for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            key = f"{stat}/ts_{family}"
            if key in prediction_metrics:
                out[f"val-core/{stat}/{display}"] = prediction_metrics[key]
        for stat in ("recall_at_1", "recall_at_5", "map"):
            for direction in ("sensor_to_text", "text_to_sensor"):
                key = f"{stat}/ts_{family}_{direction}"
                if key in prediction_metrics:
                    out[f"val-core/{stat}/{display}_{direction}"] = prediction_metrics[key]
        for stat in ("recall_at_1", "recall_at_5", "recall_at_1_macro", "recall_at_5_macro"):
            key = f"{stat}/ts_{family}"
            if key in prediction_metrics:
                out[f"val-core/{stat}/{display}"] = prediction_metrics[key]
        for source, target in (
            ("eff_dim", "effective_dimension"),
            ("prediction_coverage", "prediction_coverage"),
            ("label_coverage", "label_coverage"),
            ("class_coverage", "class_coverage"),
        ):
            key = f"{source}/ts_{family}"
            if key in prediction_metrics:
                out[f"val-aux/{target}/{display}"] = prediction_metrics[key]
        out[f"val-core/n/{display}"] = float(bucket_totals[f"n/ts_{family}"])

    for stat in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        key = f"{stat}/ts_supervised_family_macro"
        if key in prediction_metrics:
            out[f"val-core/{stat}/overall"] = prediction_metrics[key]
    for stat in ("precision_macro", "recall_macro", "f1_macro"):
        key = f"{stat}/ts_supervised_class_macro"
        if key in prediction_metrics:
            out[f"val-core/{stat}/all_classes"] = prediction_metrics[key]
    for stat in ("recall_at_1_macro", "recall_at_5_macro"):
        family_key = f"{stat}/ts_supervised_family_macro"
        class_key = f"{stat}/ts_supervised_class_macro"
        if family_key in prediction_metrics:
            out[f"val-core/{stat}/overall"] = prediction_metrics[family_key]
        if class_key in prediction_metrics:
            out[f"val-core/{stat}/all_classes"] = prediction_metrics[class_key]

    for source, target in (
        ("accuracy/smell_sensor_to_gcms", "smellnet_sensor_to_gcms"),
        ("accuracy/smell_gcms_to_text", "smellnet_gcms_to_text"),
    ):
        if source in prediction_metrics:
            out[f"val-aux/accuracy/{target}"] = prediction_metrics[source]
    return out
