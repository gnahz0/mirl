# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""SigLIP and image-preservation losses for Stage-1 alignment."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def siglip_sigmoid(
    a: torch.Tensor,
    b: torch.Tensor,
    log_logit_scale: torch.Tensor,
    bias: torch.Tensor,
    pos_mask: torch.Tensor,
    pair_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-balanced SigLIP loss over normalized embeddings."""
    a = a.float()
    b = b.float()
    scale = log_logit_scale.float().clamp(max=math.log(100.0)).exp()
    logits = scale * (a @ b.t()) + bias.float()
    z = torch.where(pos_mask, 1.0, -1.0).to(dtype=logits.dtype)
    pair_loss = -F.logsigmoid(z * logits)
    if pair_weight is None:
        return pair_loss.mean()
    weight = torch.broadcast_to(pair_weight.to(device=logits.device, dtype=logits.dtype), logits.shape)
    return (pair_loss * weight).sum() / weight.sum()


def distill_cosine(
    student: torch.Tensor,
    teacher: torch.Tensor,
    rows_per_sample: list[int] | None = None,
) -> torch.Tensor:
    """Cosine distance to a frozen teacher, optionally balanced by sample."""
    student = F.normalize(student.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher.detach().float(), dim=-1, eps=1e-6)
    row_loss = 1.0 - (student * teacher).sum(dim=-1)
    if rows_per_sample is None:
        return row_loss.mean()
    return torch.stack([chunk.mean() for chunk in row_loss.split(rows_per_sample)]).mean()
