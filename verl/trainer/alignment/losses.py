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
) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE.

    Args:
        a: (B, D), L2-normalized.
        b: (B, D), L2-normalized.
        log_logit_scale: learnable scalar tensor; the actual scale is ``exp(log_logit_scale)``,
            clamped to <= 100 the same way OpenAI CLIP does it.

    Returns:
        scalar loss = 0.5 * (CE(a -> b) + CE(b -> a)).
    """
    if a.numel() == 0 or b.numel() == 0 or a.shape[0] != b.shape[0]:
        return a.new_zeros(())
    scale = log_logit_scale.clamp(max=math.log(100.0)).exp()
    logits_ab = scale * (a @ b.t())
    logits_ba = logits_ab.t()
    targets = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))


def distill_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE on (L2-normalized) student vs teacher embeddings; equivalent up to a constant
    to ``1 - cosine_similarity`` so it pulls student toward teacher direction."""
    if student.numel() == 0 or teacher.numel() == 0:
        return student.new_zeros(())
    return F.mse_loss(student, teacher.detach())


def distill_kl_on_sim(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL between softmax rows of student-vs-student and teacher-vs-teacher similarity matrices.

    Optional; not used by default in Stage 1. Useful when you want the *distribution*
    of pairwise similarities to match rather than the raw vectors.
    """
    if student.numel() == 0 or teacher.numel() == 0 or student.shape[0] < 2:
        return student.new_zeros(())
    b = student.shape[0]
    s_sim = (student @ student.t()) / temperature
    t_sim = (teacher.detach() @ teacher.detach().t()) / temperature
    # Drop the diagonal so each row is a distribution over the B-1 *other* samples.
    # We avoid masked_fill(-inf) because the diagonal would then contribute 0 * (-inf) = NaN
    # inside the KL term.
    mask = ~torch.eye(b, device=s_sim.device, dtype=torch.bool)
    s_off = s_sim[mask].view(b, b - 1)
    t_off = t_sim[mask].view(b, b - 1)
    log_s = F.log_softmax(s_off, dim=-1)
    p_t = F.softmax(t_off, dim=-1)
    return F.kl_div(log_s, p_t, reduction="batchmean")
