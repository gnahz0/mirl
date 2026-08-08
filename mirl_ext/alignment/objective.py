# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Structured tactile SigLIP objective and evaluation."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .metrics import (
    _allreduce_counts,
    _allreduce_metrics,
    add_batch_counts,
    metric_groups,
    task_prediction_metrics,
)
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")

LabelBank = dict[str, tuple[tuple[str, ...], torch.Tensor, float]]
TaskEval = tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]


@torch.no_grad()
def build_text_label_bank(
    model: MultimodalAlignmentModel,
    task_labels: dict[str, tuple[str, ...]],
    positive_rates: dict[str, float],
    device: torch.device,
) -> LabelBank:
    bank: LabelBank = {}
    for task, labels in task_labels.items():
        text_embeddings = model.encode_text(list(labels), device=device).float().detach()
        positive_rate = positive_rates[task]
        bias = math.log(positive_rate / (1 - positive_rate))
        bank[task] = (labels, text_embeddings, bias)
        logger.info(
            "text label bank: task=%s labels=%d positive_rate=%.4f bias=%.4f",
            task,
            len(labels),
            positive_rate,
            bias,
        )
    return bank


@torch.no_grad()
def score_tasks(
    embeddings: list[torch.Tensor],
    target_chunks: dict[str, list[torch.Tensor]],
    mask_chunks: dict[str, list[torch.Tensor]],
    label_bank: LabelBank,
    log_logit_scale: torch.Tensor,
    *,
    world_size: int = 1,
    per_label_out: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    if not embeddings:
        return {}
    return task_prediction_metrics(
        torch.cat(embeddings),
        {task: torch.cat(chunks) for task, chunks in target_chunks.items()},
        {task: torch.cat(chunks) for task, chunks in mask_chunks.items()},
        label_bank,
        log_logit_scale,
        world_size=world_size,
        per_label_out=per_label_out,
    )


@torch.no_grad()
def run_validation(
    model: MultimodalAlignmentModel,
    val_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
    label_bank: LabelBank,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    was_training = model.training
    model.eval()
    device = accelerator.device
    world_size = accelerator.num_processes

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    embeddings: list[torch.Tensor] = []
    target_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    mask_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    sample_counts: Counter[str] = Counter()
    for batch in val_loader:
        add_batch_counts(sample_counts, batch)
        with accelerator.autocast():
            _, metrics, task_eval = compute_losses(model, batch, cfg, label_bank=label_bank)
        z, targets, masks = task_eval
        embeddings.append(z)
        for task in label_bank:
            target_chunks[task].append(targets[task])
            mask_chunks[task].append(masks[task])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    averaged = _allreduce_metrics(
        {key: sums[key] / counts[key] for key in sums},
        device,
        world_size,
    )
    sample_counts = _allreduce_counts(sample_counts, device, world_size)
    per_label: list[dict[str, object]] = []
    base_model = accelerator.unwrap_model(model)
    predictions = score_tasks(
        embeddings,
        target_chunks,
        mask_chunks,
        label_bank,
        base_model.log_logit_scale,
        world_size=world_size,
        per_label_out=per_label,
    )

    if was_training:
        model.train()
        base_model.label_text_model.eval()
    return metric_groups("val", averaged, sample_counts, predictions), per_label


def task_siglip_loss(
    z: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    bias: float,
    log_logit_scale: torch.Tensor,
) -> torch.Tensor | None:
    mask = mask.to(z.device)
    if not mask.any():
        return None
    target = targets.to(z.device)[mask]
    logits = log_logit_scale.float().exp() * (z[mask].float() @ text_embeddings.to(z.device).float().t()) + bias
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    *,
    label_bank: LabelBank,
) -> tuple[torch.Tensor, dict[str, float], TaskEval]:
    features, log_logit_scale = model(batch["media"])
    z = F.normalize(features.float(), dim=-1, eps=1e-6)
    task_losses: dict[str, torch.Tensor] = {}
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for task, (_, text_embeddings, bias) in label_bank.items():
        target = batch["targets"][task].to(z.device)
        mask = batch["masks"][task].to(z.device)
        loss = task_siglip_loss(
            z,
            target,
            mask,
            text_embeddings,
            bias,
            log_logit_scale,
        )
        if loss is not None:
            task_losses[task] = loss
        targets[task] = target.detach()
        masks[task] = mask.detach()

    siglip = torch.stack(tuple(task_losses.values())).mean()
    total = float(cfg.loss.siglip_weight) * siglip
    metrics = {
        "loss/siglip": siglip.detach().item(),
        "loss/total": total.detach().item(),
        "logit_scale": log_logit_scale.detach().exp().item(),
        **{f"loss/task/{task}": loss.detach().item() for task, loss in task_losses.items()},
    }
    return total, metrics, (z.detach(), targets, masks)
