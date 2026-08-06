# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Qwen3.5 vision alignment model with a frozen SigLIP2 text tower."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TACTILE_PRESSURE_MAX = 3072.0


def _resolve_snapshot(path_or_repo: str) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo)).resolve()


def _load_safetensor_prefix(
    root: Path,
    prefix: str,
    *,
    strip_prefix: bool,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted({name for key, name in weight_map.items() if key.startswith(prefix)})
    elif (root / "model.safetensors").exists():
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint found under {root}")

    state: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix) if strip_prefix else key] = handle.get_tensor(key)
    return state


def _load_exact_qwen35_visual(
    path_or_repo: str,
    *,
    dtype: torch.dtype,
    attn_impl: str,
) -> nn.Module:
    """Load the native Qwen3.5 vision tower without materializing the 9B LM."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    root = _resolve_snapshot(path_or_repo)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    vision_config = full_config.vision_config
    vision_config._attn_implementation = attn_impl
    visual = Qwen3_5VisionModel(vision_config).to(dtype=dtype)

    state = _load_safetensor_prefix(root, "model.visual.", strip_prefix=True)
    visual.load_state_dict(state, strict=True)
    return visual


def _load_exact_siglip2_text(path_or_repo: str, *, dtype: torch.dtype) -> nn.Module:
    """Load only ``text_model.*`` from the full SigLIP2 checkpoint, strictly."""
    from transformers import AutoConfig, Siglip2TextModel

    root = _resolve_snapshot(path_or_repo)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    text_model = Siglip2TextModel(full_config.text_config).to(dtype=dtype)

    state = _load_safetensor_prefix(root, "text_model.", strip_prefix=False)
    text_model.load_state_dict(state, strict=True)
    return text_model


def _enable_block_checkpointing(visual: nn.Module) -> int:
    """Checkpoint trainable vision blocks; Qwen's vision loop ignores HF's flag."""
    import torch.utils.checkpoint as checkpoint

    for block in visual.blocks:
        original_forward = block.forward

        def forward(*args, _block=block, _forward=original_forward, **kwargs):
            if _block.training:
                return checkpoint.checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        block.forward = forward
    return len(visual.blocks)


class MultimodalAlignmentModel(nn.Module):
    """Trainable Qwen vision tower, frozen image anchor, and SigLIP2 text tower."""

    def __init__(
        self,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
        visual_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        gradient_checkpointing: bool = False,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoTokenizer

        logger.info("[1/4] loading Qwen3.5 processor from %s", qwen35_path)
        started = time.time()
        qwen_root = _resolve_snapshot(qwen35_path)
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_root, local_files_only=True)
        logger.info("       processor ready (%.1fs)", time.time() - started)

        logger.info("[2/4] loading exact Qwen3.5 model.visual weights (dtype=%s, attn=%s)", visual_dtype, attn_impl)
        started = time.time()
        self.trainable_visual = _load_exact_qwen35_visual(
            str(qwen_root),
            dtype=visual_dtype,
            attn_impl=attn_impl,
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
        self.label_text_model = _load_exact_siglip2_text(str(siglip_root), dtype=visual_dtype)
        self.label_text_model.requires_grad_(False).eval()
        label_hidden = self.label_text_model.config.projection_size
        logger.info(
            "       SigLIP2 text ready: hidden=%d (%.1fs)",
            label_hidden,
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
        raw = torch.nan_to_num(signal.float())
        value = raw.clamp(-4.0, 4.0) / 4.0 if prestandardized else self._robust_normalize_rows(raw)
        value = value.masked_fill(~finite, -1.0)

        channels, steps = value.shape
        frame_count = math.ceil(steps / cell)
        padded = F.pad(value, (0, frame_count * cell - steps), value=-1.0)
        tiles = padded.reshape(channels, frame_count, cell).permute(1, 0, 2)
        tiles = tiles.repeat_interleave(cell, dim=1)
        return tiles.unsqueeze(1).expand(-1, 3, -1, -1)

    def _tactile_frame_tiles(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build ``(T,3,S,2S)`` tactile/force tiles at merger-cell resolution."""
        side = self.vit_patch_size * self.vit_merge_size
        tac = payload["tactile"]
        force = payload.get("force")
        finite = torch.isfinite(tac)
        value = torch.nan_to_num(tac.float()).clamp(0.0, _TACTILE_PRESSURE_MAX)
        value = value.mul(2.0 / _TACTILE_PRESSURE_MAX).sub(1.0)
        value = value.masked_fill(~finite, -1.0)
        value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest").squeeze(1)

        frame_count = value.shape[0]
        delta = torch.zeros_like(value)
        delta[1:] = (value[1:] - value[:-1]) * 0.5
        tactile_frames = torch.stack((value, delta, value), dim=1)

        # The adjacent merger cell carries the right-hand force summaries.
        force_frames = value.new_full((frame_count, 3, side, side), -1.0)
        if force is not None and force.numel() > 0:
            force = force.float()
            force_finite = torch.isfinite(force)
            force_raw = torch.nan_to_num(force)
            force_value = self._robust_normalize_rows(force_raw.t()).t()
            force_value = force_value.masked_fill(~force_finite, -1.0)
            num_force = force.shape[1]
            for channel in range(num_force):
                row_start = channel * side // num_force
                row_end = (channel + 1) * side // num_force
                encoded = force_value[:, channel, None, None, None]
                force_frames[:, :, row_start:row_end] = encoded.expand(-1, 3, row_end - row_start, side)

        return torch.cat((tactile_frames, force_frames), dim=-1)

    def encode_ts_trainable(
        self,
        signals: list[torch.Tensor | dict[str, torch.Tensor]],
        formats: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Render mixed native-shape signals and encode them in one vision pass."""
        videos = []
        for sig, fmt in zip(signals, formats, strict=True):
            s = (
                {key: value.to(device=device) for key, value in sig.items()}
                if isinstance(sig, dict)
                else sig.to(device=device)
            )
            if fmt == "tactile":
                frames = self._tactile_frame_tiles(s)
            else:
                frames = self._timeseries_frames(s, prestandardized=fmt == "ecg")
            videos.append(frames)

        # Sensor frames already have merger-aligned geometry and normalized values.
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
