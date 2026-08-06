# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Qwen3.5 vision alignment model with a frozen SigLIP2 text tower."""

from __future__ import annotations

import copy
import logging
import math
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _resolve_snapshot(path_or_repo: str) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo)).resolve()


def _load_exact_qwen35_visual(
    path_or_repo: str,
    *,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the native Qwen3.5 vision tower without materializing the 9B LM."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    root = _resolve_snapshot(path_or_repo)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    vision_config = full_config.vision_config
    vision_config._attn_implementation = "sdpa"
    return Qwen3_5VisionModel.from_pretrained(
        root,
        config=vision_config,
        dtype=dtype,
        local_files_only=True,
        key_mapping={r"^model\.visual\.": ""},
    )


def _enable_block_checkpointing(visual: nn.Module) -> int:
    """Wrap Qwen vision blocks because its forward loop ignores the HF flag."""
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
    """Trainable Qwen vision tower, frozen image anchor, and SigLIP2 text tower."""

    def __init__(
        self,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
        visual_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoTokenizer, Siglip2TextModel

        logger.info("[1/4] loading Qwen3.5 processor from %s", qwen35_path)
        started = time.time()
        qwen_root = _resolve_snapshot(qwen35_path)
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_root, local_files_only=True)
        logger.info("       processor ready (%.1fs)", time.time() - started)

        logger.info("[2/4] loading exact Qwen3.5 model.visual weights (dtype=%s)", visual_dtype)
        started = time.time()
        self.trainable_visual = _load_exact_qwen35_visual(
            str(qwen_root),
            dtype=visual_dtype,
        )
        logger.info(
            "       trainable VE ready: %.1fM params (%.1fs)",
            sum(p.numel() for p in self.trainable_visual.parameters()) / 1e6,
            time.time() - started,
        )

        logger.info("[3/4] cloning frozen reference vision encoder (deepcopy on CPU)")
        started = time.time()
        self.frozen_visual = copy.deepcopy(self.trainable_visual)
        self.frozen_visual.requires_grad_(False).eval()
        self.trainable_visual.merger.requires_grad_(False)
        logger.info("       frozen VE ready (%.1fs)", time.time() - started)

        if gradient_checkpointing:
            wrapped = _enable_block_checkpointing(self.trainable_visual)
            logger.info("       activation checkpointing ON: wrapped %d trainable VE blocks", wrapped)

        vcfg = self.trainable_visual.config
        self.vit_patch_size = int(vcfg.patch_size)
        self.vit_merge_size = int(vcfg.spatial_merge_size)
        logger.info(
            "[ts] signal-video formatting: patch_size=%d merge_size=%d temporal_patch_size=%d",
            self.vit_patch_size,
            self.vit_merge_size,
            int(vcfg.temporal_patch_size),
        )

        logger.info("[4/4] loading SigLIP2 label-text encoder %s", siglip2_text_path)
        started = time.time()
        siglip_root = _resolve_snapshot(siglip2_text_path)
        self.label_tokenizer = AutoTokenizer.from_pretrained(siglip_root, local_files_only=True)
        # The model class loads the text prefix and ignores the vision weights.
        self.label_text_model = Siglip2TextModel.from_pretrained(
            siglip_root, local_files_only=True
        ).to(dtype=visual_dtype)
        self.label_text_model.requires_grad_(False).eval()
        logger.info(
            "       SigLIP2 text ready: hidden=%d (%.1fs)",
            self.label_text_model.config.projection_size,
            time.time() - started,
        )

        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / contrastive_temperature)))

    def encode_visual(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        frozen: bool = False,
        pool: bool = True,
    ) -> torch.Tensor:
        visual = self.frozen_visual if frozen else self.trainable_visual
        parameter = next(visual.parameters())
        pixel_values = pixel_values.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad() if frozen else nullcontext():
            embeds = visual(pixel_values, grid_thw=grid_thw).last_hidden_state
        if not pool:
            return embeds
        counts = grid_thw.prod(dim=1).tolist()
        return torch.stack([tokens.mean(dim=0) for tokens in embeds.split(counts)])

    def forward(
        self,
        kind: str,
        media: list,
        family: str | None,
        max_image_tokens: int,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Encode one source-homogeneous batch through the trainable tower."""
        device = next(self.trainable_visual.parameters()).device
        logit_scale = self.log_logit_scale * 1.0
        if kind == "signal":
            signal_features = self.encode_ts_trainable(media, family, device)
            return None, None, None, signal_features, logit_scale

        if kind == "image":
            token_pixels = (self.vit_patch_size * self.vit_merge_size) ** 2
            max_pixels = max_image_tokens * token_pixels
            min_pixels = min(int(self.qwen_processor.image_processor.size["shortest_edge"]), max_pixels)
            processed = self.qwen_processor.image_processor.preprocess(
                media,
                size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                return_tensors="pt",
            )
            pixel_key, grid_key = "pixel_values", "image_grid_thw"
        else:
            tensors, metadata = zip(*media, strict=True)
            processed = self.qwen_processor.video_processor.preprocess(
                list(tensors),
                video_metadata=list(metadata),
                do_sample_frames=False,
                return_tensors="pt",
            )
            pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"

        pixels = processed[pixel_key].to(device=device)
        grid = processed[grid_key].to(device=device)
        token_counts = grid.prod(dim=1)
        features = self.encode_visual(pixels, grid, pool=False)
        references = self.encode_visual(pixels, grid, frozen=True, pool=False)
        return features, references, token_counts, None, logit_scale

    @staticmethod
    def _robust_normalize_rows(x: torch.Tensor) -> torch.Tensor:
        """Normalize each row with a median/MAD/std blend into ``[-1, 1]``."""
        x = torch.nan_to_num(x.float())
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values / 0.6745
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        mad_blend, tanh_gain = 0.7, 2.0
        scale = (mad_blend * mad + (1.0 - mad_blend) * std).clamp_min(1e-6)
        return torch.tanh(centered / (tanh_gain * scale))

    def _timeseries_frames(self, signal: torch.Tensor, prestandardized: bool = False) -> torch.Tensor:
        """Pack consecutive merger-cell-width signal blocks as video frames."""
        cell = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(signal)
        raw = signal.float()
        value = (
            torch.nan_to_num(raw).clamp(-4.0, 4.0) / 4.0
            if prestandardized
            else self._robust_normalize_rows(raw)
        )
        value = value.masked_fill(~finite, -1.0)

        channels, steps = value.shape
        frame_count = math.ceil(steps / cell)
        padded = F.pad(value, (0, frame_count * cell - steps), value=-1.0)
        tiles = padded.reshape(channels, frame_count, cell).permute(1, 0, 2)
        tiles = tiles.repeat_interleave(cell, dim=1)
        return tiles.unsqueeze(1).expand(-1, 3, -1, -1)

    def _tactile_frames(self, tactile: torch.Tensor) -> torch.Tensor:
        """Normalize each taxel over time and resize each map to one merger cell."""
        side = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(tactile)
        taxels = tactile.float().flatten(1).t()
        value = self._robust_normalize_rows(taxels).t().reshape_as(tactile)
        value = value.masked_fill(~finite, -1.0)
        value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest")
        return value.expand(-1, 3, -1, -1)

    def encode_ts_trainable(
        self,
        signals: list[torch.Tensor],
        family: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Render one homogeneous sensor-family batch and encode it as video."""
        if family == "tactile":
            videos = [self._tactile_frames(signal.to(device)) for signal in signals]
        else:
            videos = [
                self._timeseries_frames(signal.to(device), prestandardized=family == "ecg")
                for signal in signals
            ]

        # Sensor frames are already normalized and merger-aligned.
        processed = self.qwen_processor.video_processor.preprocess(
            videos,
            do_convert_rgb=False,
            do_sample_frames=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values_videos"].to(device=device)
        grid_thw = processed["video_grid_thw"].to(device=device)
        return self.encode_visual(pixel_values, grid_thw)

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode one max-length-truncated SigLIP2 input per text."""
        tokens = self.label_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=int(self.label_text_model.config.max_position_embeddings),
            return_tensors="pt",
        )
        embeddings = self.label_text_model(**tokens.to(device)).pooler_output.float()
        return F.normalize(embeddings, dim=-1, eps=1e-6)
