# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Qwen3.5 tactile encoder with a frozen SigLIP2 text tower."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_snapshot(path_or_repo: str) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo)).resolve()


def _load_qwen_visual(path_or_repo: str, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    root = _resolve_snapshot(path_or_repo)
    config = AutoConfig.from_pretrained(root, local_files_only=True).vision_config
    config._attn_implementation = "sdpa"
    return Qwen3_5VisionModel.from_pretrained(
        root,
        config=config,
        dtype=dtype,
        local_files_only=True,
        key_mapping={r"^model\.visual\.": ""},
    )


def _enable_block_checkpointing(visual: nn.Module) -> int:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    blocks = set(visual.blocks)
    apply_activation_checkpointing(
        visual,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=blocks.__contains__,
    )
    return len(blocks)


class MultimodalAlignmentModel(nn.Module):
    def __init__(
        self,
        qwen35_path: str,
        siglip2_text_path: str,
        visual_dtype: torch.dtype,
        gradient_checkpointing: bool,
        contrastive_temperature: float,
        max_tokens_per_sample: int,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoTokenizer, Siglip2TextModel

        qwen_root = _resolve_snapshot(qwen35_path)
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_root, local_files_only=True)
        self.trainable_visual = _load_qwen_visual(str(qwen_root), visual_dtype)
        self.trainable_visual.merger.requires_grad_(False)
        if gradient_checkpointing:
            _enable_block_checkpointing(self.trainable_visual)

        config = self.trainable_visual.config
        merger_cell = int(config.patch_size) * int(config.spatial_merge_size)
        self.frame_side = 2 * merger_cell
        spatial_tokens = (self.frame_side // merger_cell) ** 2
        self.max_frames = (
            max_tokens_per_sample // spatial_tokens * int(config.temporal_patch_size)
        )
        siglip_root = _resolve_snapshot(siglip2_text_path)
        self.label_tokenizer = AutoTokenizer.from_pretrained(siglip_root, local_files_only=True)
        self.label_text_model = Siglip2TextModel.from_pretrained(
            siglip_root,
            local_files_only=True,
        ).to(dtype=visual_dtype)
        self.label_text_model.requires_grad_(False).eval()
        self.log_logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / contrastive_temperature))
        )

    @staticmethod
    def _robust_normalize_rows(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float())
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values / 0.6745
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        scale = torch.where(mad > 1e-6, 0.7 * mad + 0.3 * std, std).clamp_min(1e-6)
        return torch.tanh(centered / (2.0 * scale))

    def _tactile_frames(self, tactile: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(tactile)
        taxels = tactile.float().flatten(1).t()
        value = self._robust_normalize_rows(taxels).t().reshape_as(tactile)
        value = value.masked_fill(~finite, -1.0)
        value = F.interpolate(
            value.unsqueeze(1),
            size=(self.frame_side, self.frame_side),
            mode="nearest",
        )
        return value.expand(-1, 3, -1, -1)

    def forward(
        self,
        recordings: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.trainable_visual.parameters())
        videos = [
            self._tactile_frames(recording[: self.max_frames].to(parameter.device))
            for recording in recordings
        ]
        processed = self.qwen_processor.video_processor.preprocess(
            videos,
            do_convert_rgb=False,
            do_sample_frames=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        pixels = processed["pixel_values_videos"].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        grid = processed["video_grid_thw"].to(parameter.device)
        tokens = self.trainable_visual(pixels, grid_thw=grid).last_hidden_state
        counts = grid.prod(dim=1).tolist()
        pooled = torch.stack([sample.mean(dim=0) for sample in tokens.split(counts)])
        return pooled, self.log_logit_scale * 1.0

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.label_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=int(self.label_text_model.config.max_position_embeddings),
            return_tensors="pt",
        )
        embeddings = self.label_text_model(**tokens.to(device)).pooler_output.float()
        return F.normalize(embeddings, dim=-1)
