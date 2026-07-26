# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Contrastive (InfoNCE) and distillation losses for Stage 1 alignment.

All inputs to ``info_nce_symmetric`` are expected to be L2-normalized along the last dim.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def info_nce_symmetric(
    a: torch.Tensor,
    b: torch.Tensor,
    log_logit_scale: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric image-text InfoNCE.

    Args:
        a: (B, D), L2-normalized.
        b: (B, D), L2-normalized.
        log_logit_scale: learnable scalar tensor; the actual scale is ``exp(log_logit_scale)``,
            clamped to <= 100 for stability.
        pos_mask: optional ``(B, B)`` boolean matrix where ``pos_mask[i, j]`` is True when row
            ``j`` is a valid positive for row ``i`` (e.g. identical text label). When given, the
            loss uses a soft target uniform over each row's positives instead of the strict
            diagonal -- this prevents duplicate labels (e.g. ECG has only 7 classes, smellnet
            repeats labels) from being treated as false negatives. The diagonal is always a
            positive. Assumed symmetric, so the same mask is used for both directions.

    Returns:
        scalar loss = 0.5 * (CE(a -> b) + CE(b -> a)).
    """
    if a.numel() == 0 or b.numel() == 0 or a.shape[0] != b.shape[0]:
        return a.new_zeros(())
    scale = log_logit_scale.clamp(max=math.log(100.0)).exp()
    logits_ab = scale * (a @ b.t())
    logits_ba = logits_ab.t()
    if pos_mask is None:
        targets = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))
    # Soft targets: distribute probability uniformly over each row's positives.
    target = pos_mask.to(dtype=logits_ab.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _soft_ce(logits: torch.Tensor) -> torch.Tensor:
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    return 0.5 * (_soft_ce(logits_ab) + _soft_ce(logits_ba))


def distill_cosine(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Cosine-distance distillation: ``mean(1 - cos(student, teacher))`` on L2-normalized
    embeddings, pulling the student toward the teacher direction.

    NOTE: this is intentionally NOT ``F.mse_loss``. Element-wise MSE with mean reduction
    divides ``||s - t||^2`` by the feature dim ``D`` (4096 for the Qwen3.5 VE), so the
    loss is ``2(1-cos)/D ~ 1e-5`` -- ~D times smaller than the O(1) InfoNCE term, leaving
    distillation with effectively zero gradient. ``1 - cos`` keeps the value/gradient O(1)
    so ``loss_weights.distill_img`` is a meaningful learn-vs-preserve knob.
    """
    if student.numel() == 0 or teacher.numel() == 0:
        return student.new_zeros(())
    return (1.0 - (student * teacher.detach()).sum(dim=-1)).mean()
