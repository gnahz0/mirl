# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 alignment losses and text-label banks."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig

from .data import TACTILE_SPANS, TASK_LABELS
from .metrics import _TS_FAMILIES
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")

LabelBank = dict[str, tuple[tuple[str, ...], torch.Tensor]]
TSEval = tuple[
    torch.Tensor | None,
    list[str],
    list[str],
    torch.Tensor | None,
    torch.Tensor | None,
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
def _build_tactile_bank(
    model: MultimodalAlignmentModel,
    device: torch.device,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Encode all six tactile task vocabularies as one bank."""
    labels: list[str] = []
    for task, task_labels in TASK_LABELS.items():
        labels.extend(task_labels)
        logger.info(
            "tactile bank: task=%s cols=%s labels=%d",
            task,
            TACTILE_SPANS[task],
            len(task_labels),
        )
    embeddings = model.encode_text(labels, device=device).float().detach()
    return tuple(labels), embeddings


def _label_siglip_loss(
    z_ts: torch.Tensor,
    labels: list[str],
    candidate_labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    log_logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
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
    logits = log_logit_scale.float().exp() * (z_ts.float() @ text_embeddings.float().t()) + logit_bias.float()
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


def _tactile_siglip_loss(
    z_ts: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    log_logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    world_size: int = 1,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Score all six tactile tasks in one masked pass over the 30-label bank.

    The tasks are independent Bernoulli decisions under the sigmoid objective, so
    they share one bank and one matmul; ``mask`` carries which ``(row, label)``
    entries have a known answer. Rows whose QA join was incomplete keep an all-zero
    target, and a zero weight is what stops that from being read as "every finger
    is absent" -- see ``_annotation_targets`` for why those rows exist at all.

    The reduction is unchanged from the per-task formulation this replaced: each
    task is meaned over its own known pairs and the tasks are then weighted equally.
    Only the plumbing collapsed -- one bank, one target/mask pair, one collective.
    """
    device = z_ts.device
    target = targets.to(device)
    weight = mask.to(device)
    logits = (
        log_logit_scale.float().exp() * (z_ts.float() @ text_embeddings.to(device).float().t()) + logit_bias.float()
    )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        weight=weight,
        reduction="none",
    )

    # One collective for the whole family. Every rank runs it unconditionally and
    # with the same shape, so it cannot desync the way a per-task reduce could.
    per_task_sum = torch.stack(
        [elementwise[:, start:stop].sum() for start, stop in TACTILE_SPANS.values()]
    )
    per_task_observed = torch.stack(
        [weight[:, start:stop].sum() for start, stop in TACTILE_SPANS.values()]
    )
    packed = torch.cat((per_task_sum.detach(), per_task_observed))
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    task_sums, task_observed = packed.reshape(2, len(TACTILE_SPANS))

    present = task_observed > 0
    if not bool(present.any()):
        return None, {}
    # Per-task mean over its own known pairs, then an equal mean across the tasks
    # that appeared. DDP averages rank gradients, so the local sum is scaled back up
    # by world_size against the globally reduced pair count.
    task_losses = per_task_sum[present] * world_size / task_observed[present]
    loss = task_losses.mean()
    per_task = {
        f"loss/task/{task}": float(task_sums[index] / task_observed[index])
        for index, task in enumerate(TACTILE_SPANS)
        if bool(present[index])
    }
    return loss, per_task


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    *,
    label_bank: LabelBank,
    tactile_bank: tuple[tuple[str, ...], torch.Tensor],
    world_size: int = 1,
) -> tuple[torch.Tensor, dict, TSEval]:
    """Compute family label-bank SigLIP and frozen-Qwen preservation losses."""
    metrics: dict[str, float] = {}

    kind = batch["kind"]
    media = batch["media"]
    family = batch["family"]
    labels = batch["text"]
    feat_img, feat_ref_img, img_token_counts, feat_ts, log_logit_scale, logit_bias = model(
        kind,
        media,
        family,
        int(cfg.data.max_image_tokens),
    )

    total = (log_logit_scale.float() + logit_bias.float()) * 0.0
    tactile_targets: torch.Tensor | None = None
    tactile_masks: torch.Tensor | None = None

    z_ts = F.normalize(feat_ts.float(), dim=-1, eps=1e-6) if feat_ts is not None else None
    if z_ts is not None:
        if family == "tactile":
            _, text_embeddings = tactile_bank
            tactile_targets = batch["targets"].to(z_ts.device)
            tactile_masks = batch["masks"].to(z_ts.device)
            l_ts, per_task = _tactile_siglip_loss(
                z_ts,
                tactile_targets,
                tactile_masks,
                text_embeddings,
                log_logit_scale,
                logit_bias,
                world_size,
            )
            metrics.update(per_task)
            tactile_targets = tactile_targets.detach()
            tactile_masks = tactile_masks.detach()
        else:
            candidate_labels, text_embeddings = label_bank[family]
            l_ts = _label_siglip_loss(
                z_ts,
                labels,
                candidate_labels,
                text_embeddings,
                log_logit_scale,
                logit_bias,
                world_size,
            )
        # A tactile batch whose rows are all unannotated for every task yields no
        # loss at all. Every rank sees the same batch, so they agree on the skip.
        if l_ts is not None:
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
    metrics["logit_bias"] = logit_bias.detach().item()
    ts_eval = (
        (z_ts.detach(), labels, [family] * len(labels), tactile_targets, tactile_masks)
        if z_ts is not None
        else (None, [], [], None, None)
    )
    return total, metrics, ts_eval
