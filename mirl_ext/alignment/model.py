"""Qwen3.5 vision alignment model with a frozen SigLIP2 text tower."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from mirl_ext.data import signals as _signals


class MultimodalAlignmentModel(nn.Module):
    """Trainable Qwen vision tower, frozen image anchor, and SigLIP2 text tower."""

    def __init__(
        self,
        max_video_frames: int,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Siglip2TextModel

        super().__init__()

        self.qwen_processor = AutoProcessor.from_pretrained(
            qwen35_path,
            local_files_only=True,
            fps=None,
            min_frames=1,
            max_frames=max_video_frames,
        )
        full_model = AutoModelForImageTextToText.from_pretrained(
            qwen35_path,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        # Keep only the vision tower; the language stack is freed with full_model.
        self.trainable_visual = full_model.model.visual
        del full_model

        self.frozen_visual = copy.deepcopy(self.trainable_visual)
        self.frozen_visual.requires_grad_(False).eval()
        self.trainable_visual.merger.requires_grad_(False)

        vcfg = self.trainable_visual.config
        self.vit_patch_size = int(vcfg.patch_size)
        self.vit_merge_size = int(vcfg.spatial_merge_size)

        self.label_tokenizer = AutoTokenizer.from_pretrained(siglip2_text_path, local_files_only=True)
        self.label_text_model = Siglip2TextModel.from_pretrained(siglip2_text_path, local_files_only=True).to(dtype=torch.bfloat16)
        self.label_text_model.requires_grad_(False).eval()

        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

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
        torch.Tensor,
    ]:
        """Encode one source-homogeneous batch through the trainable tower."""
        device = next(self.trainable_visual.parameters()).device
        logit_scale = self.log_logit_scale * 1.0
        logit_bias = self.logit_bias * 1.0
        if kind == "signal":
            return None, None, None, self.encode_ts_trainable(media, family, device), logit_scale, logit_bias

        if kind == "image":
            token_pixels = (self.vit_patch_size * self.vit_merge_size) ** 2
            max_pixels = max_image_tokens * token_pixels
            processed = self.qwen_processor.image_processor(
                media,
                size={**self.qwen_processor.image_processor.size, "longest_edge": max_pixels},
                return_tensors="pt",
            )
            pixel_key, grid_key = "pixel_values", "image_grid_thw"
        else:
            processed = self.qwen_processor.video_processor(media, return_tensors="pt")
            pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"

        pixels = processed[pixel_key].to(device=device)
        grid = processed[grid_key].to(device=device)
        token_counts = grid.prod(dim=1)
        features = self.encode_visual(pixels, grid, pool=False)
        references = self.encode_visual(pixels, grid, frozen=True, pool=False)
        return features, references, token_counts, None, logit_scale, logit_bias
    
    def encode_ts_trainable(
        self,
        signals: list[torch.Tensor],
        family: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Render one homogeneous sensor-family batch and encode it as video."""
        cell = self.vit_patch_size * self.vit_merge_size
        videos = [_signals.family_frames(signal.to(device), family, cell) for signal in signals]

        processed = self.qwen_processor.video_processor(
            videos,
            do_convert_rgb=False,
            do_sample_frames=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        return self.encode_visual(
            processed["pixel_values_videos"].to(device=device),
            processed["video_grid_thw"].to(device=device),
        )

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
            max_length=self.label_text_model.config.max_position_embeddings,
            return_tensors="pt",
        )
        text_embeds = self.label_text_model(**tokens.to(device)).pooler_output.float()
        return F.normalize(text_embeds, dim=-1)
