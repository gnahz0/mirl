"""Qwen3.5 vision alignment model with a frozen SigLIP2 text tower."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalAlignmentModel(nn.Module):
    """Trainable Qwen vision tower, frozen image anchor, and SigLIP2 text tower."""

    def __init__(
        self,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
    ):
        from transformers import AutoProcessor, AutoTokenizer, Siglip2TextModel
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        super().__init__()

        self.qwen_processor = AutoProcessor.from_pretrained(qwen35_path, local_files_only=True)
        self.trainable_visual = Qwen3_5VisionModel.from_pretrained(
            qwen35_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            key_mapping={r"^model\.visual\.": ""},
        )

        self.frozen_visual = copy.deepcopy(self.trainable_visual)
        self.frozen_visual.requires_grad_(False).eval()
        self.trainable_visual.merger.requires_grad_(False)

        vcfg = self.trainable_visual.config
        self.vit_patch_size = int(vcfg.patch_size)
        self.vit_merge_size = int(vcfg.spatial_merge_size)

        self.label_tokenizer = AutoTokenizer.from_pretrained(siglip2_text_path, local_files_only=True)
        self.label_text_model = Siglip2TextModel.from_pretrained(siglip2_text_path, local_files_only=True).to(
            dtype=torch.bfloat16
        )
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
        return features, references, token_counts, None, logit_scale, logit_bias

    @staticmethod
    def _robust_normalize_rows(x: torch.Tensor) -> torch.Tensor:
        """Normalize each row with a robust scale and a std fallback."""
        x = torch.nan_to_num(x.float())
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values / 0.6745
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        mad_blend, tanh_gain = 0.7, 2.0
        blended = mad_blend * mad + (1.0 - mad_blend) * std
        # Quantized/sparse traces can have MAD=0 despite variation: use full std, not the 0.3-shrunk blend.
        # Constant rows still map exactly to 0.
        scale = torch.where(mad > 1e-6, blended, std).clamp_min(1e-6)
        return torch.tanh(centered / (tanh_gain * scale))

    def _timeseries_frames(self, signal: torch.Tensor, prestandardized: bool = False) -> torch.Tensor:
        """Pack consecutive merger-cell-width signal blocks as video frames."""
        cell = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(signal)
        raw = signal.float()
        value = torch.nan_to_num(raw).clamp(-4.0, 4.0) / 4.0 if prestandardized else self._robust_normalize_rows(raw)
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
            videos = [self._timeseries_frames(signal.to(device), prestandardized=family == "ecg") for signal in signals]

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
            max_length=int(self.label_text_model.config.max_position_embeddings),
            return_tensors="pt",
        )
        embeddings = self.label_text_model(**tokens.to(device)).pooler_output.float()
        return F.normalize(embeddings, dim=-1, eps=1e-6)
