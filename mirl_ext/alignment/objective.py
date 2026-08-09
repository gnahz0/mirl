# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 alignment objective, text-label banks, and evaluation."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .metrics import (
    _TS_FAMILIES,
    _allreduce_counts,
    _allreduce_metrics,
    _merge_prediction_metrics,
    _metric_groups,
    _tactile_prediction_metrics,
    _ts_prediction_metrics,
    add_batch_counts,
)
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")

LabelBank = dict[str, tuple[tuple[str, ...], torch.Tensor]]
TactileLabelBank = dict[str, tuple[tuple[str, ...], torch.Tensor, float]]
TSEval = tuple[
    torch.Tensor | None,
    list[str],
    list[str],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]


@torch.no_grad()
def _build_text_label_bank(
    model: MultimodalAlignmentModel,
    vocabularies: dict[str, tuple[str, ...]],
    device: torch.device,
    batch_size: int = 256,
) -> LabelBank:
    """Encode each split's complete family label vocabulary once with SigLIP2."""
    bank: LabelBank = {}
    for family in _TS_FAMILIES:
        labels = tuple(vocabularies.get(family, ()))
        if not labels:
            continue
        chunks = [
            model.encode_text(list(labels[start : start + batch_size]), device=device).float()
            for start in range(0, len(labels), batch_size)
        ]
        bank[family] = (labels, torch.cat(chunks, dim=0).detach())
        logger.info("text label bank: family=%s labels=%d", family, len(labels))
    return bank


@torch.no_grad()
def _build_tactile_label_bank(
    model: MultimodalAlignmentModel,
    task_labels: dict[str, tuple[str, ...]],
    positive_rates: dict[str, float],
    device: torch.device,
) -> TactileLabelBank:
    """Encode the six structured tactile task vocabularies once."""
    bank: TactileLabelBank = {}
    for task, labels in task_labels.items():
        positive_rate = float(positive_rates[task])
        if not 0.0 < positive_rate < 1.0:
            raise ValueError(f"structured tactile positive rate must be in (0, 1): {task}={positive_rate}")
        embeddings = model.encode_text(list(labels), device=device).float().detach()
        bias = math.log(positive_rate / (1.0 - positive_rate))
        bank[task] = (labels, embeddings, bias)
        logger.info(
            "text label bank: tactile task=%s labels=%d positive_rate=%.4f bias=%.4f",
            task,
            len(labels),
            positive_rate,
            bias,
        )
    return bank


@torch.no_grad()
def _score_ts(
    embeddings: list[torch.Tensor],
    labels: list[str],
    families: list[str],
    label_bank: LabelBank,
    world_size: int = 1,
    per_class_reports: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, float]:
    """Score one sensor sample set against the frozen text-label bank."""
    if not embeddings:
        return {}
    return _ts_prediction_metrics(
        torch.cat(embeddings),
        labels,
        families,
        label_bank,
        world_size=world_size,
        per_class_reports=per_class_reports,
    )


@torch.no_grad()
def _score_tactile(
    embeddings: list[torch.Tensor],
    target_chunks: dict[str, list[torch.Tensor]],
    mask_chunks: dict[str, list[torch.Tensor]],
    label_bank: TactileLabelBank,
    log_logit_scale: torch.Tensor,
    *,
    world_size: int = 1,
    per_label_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Score structured tactile targets over one accumulation/eval window."""
    if not embeddings:
        return {}
    return _tactile_prediction_metrics(
        torch.cat(embeddings),
        {task: torch.cat(target_chunks[task]) for task in label_bank},
        {task: torch.cat(mask_chunks[task]) for task in label_bank},
        label_bank,
        log_logit_scale,
        world_size=world_size,
        per_label_out=per_label_out,
    )


@torch.no_grad()
def _run_validation(
    model: MultimodalAlignmentModel,
    val_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
    label_bank: LabelBank,
    tactile_label_bank: TactileLabelBank,
) -> tuple[
    dict[str, float],
    dict[str, list[dict[str, object]]],
    list[dict[str, object]],
]:
    """Evaluate losses and prediction metrics over validation batches."""
    was_training = model.training
    model.eval()
    device = accelerator.device
    world_size = accelerator.num_processes

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    ts_embeddings: list[torch.Tensor] = []
    ts_labels: list[str] = []
    ts_families: list[str] = []
    tactile_embeddings: list[torch.Tensor] = []
    tactile_target_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    tactile_mask_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    bucket_totals: Counter[str] = Counter()
    for batch in val_loader:
        add_batch_counts(bucket_totals, batch)
        with accelerator.autocast():
            _, metrics, batch_eval = _compute_losses(
                model,
                batch,
                cfg,
                world_size=world_size,
                label_bank=label_bank,
                tactile_label_bank=tactile_label_bank,
            )
        embeddings, labels, families, tactile_targets, tactile_masks = batch_eval
        if embeddings is not None:
            if tactile_targets:
                tactile_embeddings.append(embeddings)
                for task in tactile_label_bank:
                    tactile_target_chunks[task].append(tactile_targets[task])
                    tactile_mask_chunks[task].append(tactile_masks[task])
            else:
                ts_embeddings.append(embeddings)
                ts_labels.extend(labels)
                ts_families.extend(families)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    averaged = _allreduce_metrics(
        {key: sums[key] / counts[key] for key in sums},
        device,
        world_size,
    )
    bucket_totals = _allreduce_counts(bucket_totals, device, world_size)
    per_class_reports: dict[str, list[dict[str, object]]] = {}
    standard_metrics = _score_ts(
        ts_embeddings,
        ts_labels,
        ts_families,
        label_bank,
        world_size=world_size,
        per_class_reports=per_class_reports,
    )
    per_label_report: list[dict[str, object]] = []
    base_model = accelerator.unwrap_model(model)
    tactile_metrics = _score_tactile(
        tactile_embeddings,
        tactile_target_chunks,
        tactile_mask_chunks,
        tactile_label_bank,
        base_model.log_logit_scale,
        world_size=world_size,
        per_label_out=per_label_report,
    )
    prediction_metrics = _merge_prediction_metrics(standard_metrics, tactile_metrics)

    if was_training:
        model.train()
        base_model.frozen_visual.eval()
        base_model.label_text_model.eval()

    out = _metric_groups("val", averaged, bucket_totals, prediction_metrics)
    return out, per_class_reports, per_label_report


def _label_siglip_loss(
    z_ts: torch.Tensor,
    labels: list[str],
    candidate_labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    log_logit_scale: torch.Tensor,
    world_size: int = 1,
) -> torch.Tensor:
    """Compute class-balanced SigLIP against one complete text-label bank."""
    num_labels = len(candidate_labels)
    label_to_id = {label: idx for idx, label in enumerate(candidate_labels)}
    targets = torch.tensor(
        [label_to_id[label] for label in labels],
        device=z_ts.device,
        dtype=torch.long,
    )
    # Repeated labels contribute the same total anchor weight as unique labels.
    class_count = torch.bincount(targets, minlength=num_labels).float()
    if world_size > 1:
        dist.all_reduce(class_count, op=dist.ReduceOp.SUM)
    sample_weight = class_count.index_select(0, targets).reciprocal()
    bias = z_ts.new_tensor(-math.log(num_labels - 1))
    logits = log_logit_scale.float().exp() * (z_ts.float() @ text_embeddings.float().t()) + bias
    positives = F.one_hot(targets, num_classes=num_labels).to(dtype=logits.dtype)
    local_sum = F.binary_cross_entropy_with_logits(
        logits,
        positives,
        weight=sample_weight[:, None],
        reduction="sum",
    )
    # Match SigLIP's reduction: sum candidate-pair losses for each anchor, then
    # average anchors. Here the anchor mean is class-balanced, and DDP averages
    # rank gradients, so scale local rows back to the global supported-class mean.
    denominator = (class_count > 0).sum()
    return local_sum * world_size / denominator


def _tactile_task_siglip_loss(
    z_ts: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    bias: float,
    log_logit_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Compute one masked multi-positive SigLIP-style tactile task loss."""
    mask = mask.to(z_ts.device)
    if not mask.any():
        return None
    target = targets.to(z_ts.device)[mask]
    logits = log_logit_scale.float().exp() * (z_ts[mask].float() @ text_embeddings.to(z_ts.device).float().t()) + bias
    return F.binary_cross_entropy_with_logits(logits, target)


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    *,
    label_bank: LabelBank,
    tactile_label_bank: TactileLabelBank | None = None,
    world_size: int = 1,
) -> tuple[torch.Tensor, dict, TSEval]:
    """Compute family label-bank SigLIP and frozen-Qwen preservation losses."""
    metrics: dict[str, float] = {}

    kind = batch["kind"]
    media = batch["media"]
    family = batch["family"]
    labels = batch["text"]
    feat_img, feat_ref_img, img_token_counts, feat_ts, log_logit_scale = model(
        kind,
        media,
        family,
        int(cfg.data.max_image_tokens),
    )

    total = log_logit_scale.float() * 0.0
    tactile_targets: dict[str, torch.Tensor] = {}
    tactile_masks: dict[str, torch.Tensor] = {}

    z_ts = F.normalize(feat_ts.float(), dim=-1, eps=1e-6) if feat_ts is not None else None
    if z_ts is not None:
        if family == "tactile" and "targets" in batch:
            if not tactile_label_bank:
                raise ValueError("structured tactile batch has no tactile label bank")
            task_losses: dict[str, torch.Tensor] = {}
            for task, (_, text_embeddings, bias) in tactile_label_bank.items():
                target = batch["targets"][task].to(z_ts.device)
                mask = batch["masks"][task].to(z_ts.device)
                task_loss = _tactile_task_siglip_loss(
                    z_ts,
                    target,
                    mask,
                    text_embeddings,
                    bias,
                    log_logit_scale,
                )
                if task_loss is not None:
                    task_losses[task] = task_loss
                    metrics[f"loss/task/{task}"] = task_loss.detach().item()
                tactile_targets[task] = target.detach()
                tactile_masks[task] = mask.detach()
            if not task_losses:
                raise RuntimeError("tactile batch has no observed structured task targets")
            l_ts = torch.stack(tuple(task_losses.values())).mean()
        else:
            candidate_labels, text_embeddings = label_bank[family]
            l_ts = _label_siglip_loss(
                z_ts,
                labels,
                candidate_labels,
                text_embeddings,
                log_logit_scale,
                world_size,
            )
        total = total + float(cfg.loss.siglip_weight) * l_ts
        metrics["loss/siglip"] = l_ts.detach().item()
        metrics[f"loss/ts_{family}"] = l_ts.detach().item()

    visual_sample_loss = total.new_empty(0)
    if feat_img is not None:
        token_loss = 1.0 - F.cosine_similarity(
            feat_img.float(),
            feat_ref_img.detach().float(),
            dim=-1,
            eps=1e-6,
        )
        visual_sample_loss = torch.segment_reduce(
            token_loss,
            reduce="mean",
            lengths=img_token_counts,
        )

    visual_count = log_logit_scale.new_tensor(visual_sample_loss.numel())
    if world_size > 1:
        dist.all_reduce(visual_count, op=dist.ReduceOp.SUM)
    global_visual_count = int(visual_count.item())
    if global_visual_count:
        # DDP averages rank gradients; this preserves the exact global sample mean.
        l_img = visual_sample_loss.sum() * world_size / global_visual_count
        total = total + float(cfg.loss.distill_weight) * l_img
        metrics["loss/distill"] = l_img.detach().item()

    metrics["loss/total"] = total.detach().item()
    metrics["logit_scale"] = log_logit_scale.detach().exp().item()
    ts_eval = (
        (z_ts.detach(), labels, [family] * len(labels), tactile_targets, tactile_masks)
        if z_ts is not None
        else (None, [], [], {}, {})
    )
    return total, metrics, ts_eval
