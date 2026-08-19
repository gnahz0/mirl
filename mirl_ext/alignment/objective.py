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
    return tuple(labels), model.encode_text(labels, device=device).float().detach()


def _label_siglip_loss(
    z_ts: torch.Tensor,
    labels: list[str],
    candidate_labels: tuple[str, ...],
    text_embeddings: torch.Tensor,
    log_logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    world_size: int = 1,
) -> torch.Tensor:
    """Compute class-balanced SigLIP against one complete text-label bank.

    Args:
        z_ts: (N, D) L2-normalized signal embeddings for the rank-local rows.
        labels: Ground-truth label per row; each must be in ``candidate_labels``.
        candidate_labels: The family's complete label vocabulary.
        text_embeddings: (K, D) bank embeddings aligned with ``candidate_labels``.
        log_logit_scale: Learned SigLIP temperature (log space).
        logit_bias: Learned SigLIP bias.
        world_size: DDP world size; class counts reduce globally.

    Returns:
        Scalar loss: candidate-pair BCE summed per anchor, then a class-balanced
        anchor mean, pre-scaled by ``world_size`` so DDP's gradient averaging
        yields the exact global supported-class mean.
    """
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
    # SigLIP reduction: sum candidate-pair losses per anchor, then a class-balanced anchor mean.
    # DDP averages rank gradients, so scale local rows back to the global supported-class mean.
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
    """Score all six tactile tasks in one masked pass over the shared 30-label bank.

    ``mask`` marks known (row, label) pairs; only a zero weight stops an unannotated
    row's all-zero target from reading as "every finger is absent".

    Args:
        z_ts: (N, D) L2-normalized tactile embeddings for the rank-local rows.
        targets: (N, 30) multi-hot targets over the concatenated task spans.
        mask: (N, 30) 1.0 where the row's task is annotated, 0.0 elsewhere.
        text_embeddings: (30, D) bank from ``_build_tactile_bank``.
        log_logit_scale: Learned SigLIP temperature (log space).
        logit_bias: Learned SigLIP bias.
        world_size: DDP world size; per-task observed counts reduce globally.

    Returns:
        (loss, per_task): the equal-weight mean over globally observed task
        losses (``None`` when no task is observed on any rank), and detached
        ``loss/task/<name>`` diagnostics for the observed tasks.
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

    # One unconditional same-shape collective for the whole family: it cannot desync like a per-task reduce could.
    per_task_sum = torch.stack([elementwise[:, start:stop].sum() for start, stop in TACTILE_SPANS.values()])
    per_task_observed = torch.stack([weight[:, start:stop].sum() for start, stop in TACTILE_SPANS.values()])
    packed = torch.cat((per_task_sum.detach(), per_task_observed))
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    task_sums, task_observed = packed.reshape(2, len(TACTILE_SPANS))

    present = task_observed > 0
    if not bool(present.any()):
        return None, {}

    task_losses = per_task_sum[present] * world_size / task_observed[present]
    per_task = {
        f"loss/task/{task}": float(task_sums[index] / task_observed[index])
        for index, task in enumerate(TACTILE_SPANS)
        if bool(present[index])
    }
    return task_losses.mean(), per_task


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    *,
    label_bank: LabelBank,
    tactile_bank: tuple[tuple[str, ...], torch.Tensor],
    world_size: int = 1,
) -> tuple[torch.Tensor, dict, TSEval]:
    """Compute family label-bank SigLIP and frozen-Qwen preservation losses.

    Args:
        model: Alignment model; its forward encodes the batch's media.
        batch: One source-homogeneous batch from ``collate_alignment``.
        cfg: Run config; reads ``loss.siglip_weight``, ``loss.distill_weight``,
            and ``data.max_image_tokens``.
        label_bank: Family -> (labels, embeddings) from ``_build_text_label_bank``.
        tactile_bank: (labels, embeddings) from ``_build_tactile_bank``.
        world_size: DDP world size for the global loss reductions.

    Returns:
        (total, metrics, ts_eval): the weighted scalar training loss, detached
        scalar metrics keyed for W&B, and the ``TSEval`` tuple consumed by
        ``update_stats`` — (embeddings, labels, families, tactile targets,
        tactile masks), empty/None for visual batches.
    """
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
        # A fully-unannotated tactile batch yields no loss; every rank sees the same batch, so they agree on the skip.
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
    return (
        total,
        metrics,
        (
            (z_ts.detach(), labels, [family] * len(labels), tactile_targets, tactile_masks)
        ),
    )
