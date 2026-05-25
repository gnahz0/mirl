# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Multimodal alignment model wrapper.

Holds:
    - ``trainable_visual``  : Qwen3VLForConditionalGeneration.visual (the middle VE, trainable).
    - ``frozen_visual``     : another copy of the same VE, frozen (right-side reference VE).
    - ``clip_text_model``   : HuggingFace CLIPTextModel (frozen).
    - projection heads     : map each encoder output to a shared embedding dim.
    - ``log_logit_scale``  : learnable CLIP-style temperature.

Vision encoding details:
    Qwen3-VL's ``model.visual(pixel_values, grid_thw=image_grid_thw)`` returns
    ``(image_embeds, deepstack_embeds)`` where ``image_embeds`` has shape
    ``[sum_i n_i, hidden]`` (flat, all images concatenated). We split using the cumulative
    per-image token counts derived from ``image_grid_thw`` (each row is ``(t, h, w)``
    *post*-merge so ``n_i = t * h * w``) and mean-pool per image to ``[B, hidden]``.

TODO(stage2): export the trained ``trainable_visual.state_dict()`` back into a full
Qwen3-VL HF checkpoint and point veRL's ``actor_rollout_ref.model.path`` at it.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import ProjectionHead

logger = logging.getLogger(__name__)


def _split_and_pool(
    flat_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """flat_embeds: [sum_i n_i, hidden]; grid_thw: [B, 3] with rows (t, h, w) *post-merge*.

    Returns: [B, hidden] mean-pooled per image.
    """
    if flat_embeds.numel() == 0 or grid_thw is None or grid_thw.numel() == 0:
        return flat_embeds.new_zeros((0, flat_embeds.shape[-1] if flat_embeds.ndim >= 1 else 1))
    counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    pooled = []
    start = 0
    for c in counts:
        if c == 0:
            pooled.append(flat_embeds.new_zeros(flat_embeds.shape[-1]))
            continue
        chunk = flat_embeds[start:start + c]
        pooled.append(chunk.mean(dim=0))
        start += c
    return torch.stack(pooled, dim=0)


class MultimodalAlignmentModel(nn.Module):
    """Stage 1 wrapper.

    Args:
        qwen3_vl_path: HF id or local path for Qwen3-VL (e.g. ``Qwen/Qwen3-VL-8B-Instruct``).
        clip_text_path: HF id for the CLIP text encoder (default ``openai/clip-vit-large-patch14``).
        shared_dim: projection output dim.
        proj_hidden_dim: projection MLP hidden width.
        visual_dtype: dtype for both VEs (bf16 strongly recommended).
        attn_impl: attention implementation passed to ``from_pretrained`` for the Qwen model.
    """

    def __init__(
        self,
        qwen3_vl_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        clip_text_path: str = "openai/clip-vit-large-patch14",
        shared_dim: int = 512,
        proj_hidden_dim: Optional[int] = 1024,
        proj_dropout: float = 0.0,
        visual_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
    ):
        super().__init__()
        from transformers import (
            AutoProcessor,
            CLIPTextModel,
            CLIPTokenizer,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        # ---- Qwen3-VL processor (shared between trainable & frozen VE) ----
        self.qwen_processor = AutoProcessor.from_pretrained(qwen3_vl_path)

        # ---- Trainable VE (middle): load full model, keep only .visual to save memory ----
        logger.info("loading trainable Qwen3-VL from %s", qwen3_vl_path)
        trainable_full = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen3_vl_path, dtype=visual_dtype, attn_implementation=attn_impl,
        )
        self.trainable_visual = trainable_full.visual
        # We don't need the LM tower in Stage 1; drop it.
        del trainable_full.model
        del trainable_full.lm_head
        del trainable_full

        # ---- Frozen reference VE (right): identical weights, requires_grad=False ----
        logger.info("cloning frozen reference vision encoder")
        self.frozen_visual = copy.deepcopy(self.trainable_visual)
        for p in self.frozen_visual.parameters():
            p.requires_grad_(False)
        self.frozen_visual.eval()

        qwen_hidden = self._infer_qwen_visual_hidden(self.trainable_visual)

        # ---- CLIP text encoder (frozen) ----
        logger.info("loading CLIP text encoder %s", clip_text_path)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_text_path)
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_text_path)
        for p in self.clip_text_model.parameters():
            p.requires_grad_(False)
        self.clip_text_model.eval()
        clip_hidden = self.clip_text_model.config.hidden_size

        # ---- Projection heads ----
        self.proj_img = ProjectionHead(qwen_hidden, shared_dim, proj_hidden_dim, proj_dropout)
        self.proj_ts_img = ProjectionHead(qwen_hidden, shared_dim, proj_hidden_dim, proj_dropout)
        self.proj_ref = ProjectionHead(qwen_hidden, shared_dim, proj_hidden_dim, proj_dropout)
        self.proj_ref_ts = ProjectionHead(qwen_hidden, shared_dim, proj_hidden_dim, proj_dropout)
        self.proj_text = ProjectionHead(clip_hidden, shared_dim, proj_hidden_dim, proj_dropout)

        # ---- CLIP-style learnable temperature ----
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        self.shared_dim = shared_dim
        self.qwen_hidden = qwen_hidden
        self.clip_hidden = clip_hidden
        self.visual_dtype = visual_dtype

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_qwen_visual_hidden(visual_module: nn.Module) -> int:
        """The Qwen3-VL visual tower exposes its output dim via config.out_hidden_size
        (post merger). Fall back to inspecting the merger head if needed."""
        cfg = getattr(visual_module, "config", None)
        if cfg is not None:
            for attr in ("out_hidden_size", "hidden_size"):
                if hasattr(cfg, attr):
                    return int(getattr(cfg, attr))
        for name, mod in visual_module.named_modules():
            if isinstance(mod, nn.Linear) and "merger" in name:
                return mod.out_features
        raise RuntimeError("could not infer Qwen3-VL visual output dim")

    def trainable_parameter_groups(self, lr: float, weight_decay: float):
        """Param groups for the optimizer. The frozen VE and CLIP text encoder are excluded.
        Bias / LayerNorm / Norm params get weight_decay=0 (standard practice)."""
        decay, no_decay = [], []
        for p_name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or p_name.endswith(".bias") or "logit_scale" in p_name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    # -------------------------------------------------------------------------
    # Branch encoders
    # -------------------------------------------------------------------------

    def _encode_qwen_branch(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        visual: nn.Module,
        no_grad: bool,
    ) -> torch.Tensor:
        """Returns [B, qwen_hidden] mean-pooled per image."""
        if pixel_values.numel() == 0 or image_grid_thw.numel() == 0:
            return pixel_values.new_zeros((0, self.qwen_hidden))
        pixel_values = pixel_values.to(dtype=visual.dtype if hasattr(visual, "dtype") else self.visual_dtype)
        ctx = torch.no_grad() if no_grad else _nullcontext()
        with ctx:
            embeds, _deepstack = visual(pixel_values, grid_thw=image_grid_thw)
        return _split_and_pool(embeds, image_grid_thw)

    def encode_images_trainable(self, pixel_values, image_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values, image_grid_thw, visual=self.trainable_visual, no_grad=False
        )

    def encode_images_frozen(self, pixel_values, image_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values, image_grid_thw, visual=self.frozen_visual, no_grad=True
        )

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device, max_length: int = 77) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.clip_hidden), device=device)
        toks = self.clip_tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        out = self.clip_text_model(**toks)
        # CLIPTextModel returns ``pooler_output`` = EOS-token hidden state (CLIP standard)
        return out.pooler_output

    # -------------------------------------------------------------------------
    # Projection + normalize convenience
    # -------------------------------------------------------------------------

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if x.numel() > 0 else x

    def project(self, head: ProjectionHead, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, self.shared_dim))
        return self._norm(head(x.to(next(head.parameters()).dtype)))


class _nullcontext:
    """Tiny no-op context manager (avoids importing contextlib here)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
