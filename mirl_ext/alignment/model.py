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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import ProjectionHead

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
    """Load only ``text_model.*`` from a paired SigLIP2 checkpoint, strictly."""
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


def _split_and_pool(
    flat_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_unit: int,
) -> torch.Tensor:
    """Split post-merger tokens by ``grid_thw`` and mean-pool each sample."""
    counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2] // merge_unit).tolist()
    if sum(counts) != flat_embeds.shape[0]:
        raise ValueError(
            f"VE output rows ({flat_embeds.shape[0]}) != sum of per-sample post-merge "
            f"counts ({sum(counts)}); grid_thw={grid_thw.tolist()}, merge_unit={merge_unit}"
        )
    pooled = []
    start = 0
    for c in counts:
        chunk = flat_embeds[start : start + c]
        pooled.append(chunk.mean(dim=0))
        start += c
    return torch.stack(pooled, dim=0)


class MultimodalAlignmentModel(nn.Module):
    """Trainable Qwen vision tower, frozen image anchor, and SigLIP2 text tower."""

    def __init__(
        self,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
        shared_dim: int = 512,
        visual_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        gradient_checkpointing: bool = False,
        ecg_normalization: str = "robust",
        tactile_delta_channels: bool = False,
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
        logger.info("       frozen VE ready (%.1fs)", time.time() - started)

        if gradient_checkpointing:
            wrapped = _enable_block_checkpointing(self.trainable_visual)
            logger.info("       activation checkpointing ON: wrapped %d trainable VE blocks", wrapped)

        qwen_hidden = int(self.trainable_visual.config.out_hidden_size)

        vcfg = self.trainable_visual.config
        self.vit_patch_size = int(vcfg.patch_size)
        self.vit_merge_size = int(vcfg.spatial_merge_size)
        self.vit_temporal_patch_size = int(vcfg.temporal_patch_size)
        logger.info(
            "[ts] signal-video formatting: patch_size=%d merge_size=%d temporal_patch_size=%d",
            self.vit_patch_size,
            self.vit_merge_size,
            self.vit_temporal_patch_size,
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

        self.proj_visual = ProjectionHead(qwen_hidden, shared_dim)
        self.proj_text = ProjectionHead(label_hidden, shared_dim)

        self.tactile_delta_channels = bool(tactile_delta_channels)
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / contrastive_temperature)))

        self.shared_dim = shared_dim
        self.ecg_normalization = ecg_normalization

    def trainable_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        head_lr: Optional[float] = None,
        scalar_lr: Optional[float] = None,
    ):
        """Build ViT, projection-head, and scalar optimizer tiers."""
        head_lr = lr if head_lr is None else head_lr
        scalar_lr = head_lr if scalar_lr is None else scalar_lr
        groups = {
            ("vit", "decay"): [],
            ("vit", "no_decay"): [],
            ("head", "decay"): [],
            ("head", "no_decay"): [],
            ("scalar", "no_decay"): [],
        }
        for p_name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 0:
                groups[("scalar", "no_decay")].append(p)
                continue
            tier = "vit" if p_name.startswith("trainable_visual.") else "head"
            no_decay = p.ndim <= 1 or p_name.endswith(".bias")
            groups[(tier, "no_decay" if no_decay else "decay")].append(p)
        lr_for = {"vit": lr, "head": head_lr, "scalar": scalar_lr}
        out = []
        for (tier, kind), params in groups.items():
            if not params:
                continue
            out.append(
                {
                    "name": f"{tier}_{kind}",
                    "params": params,
                    "lr": lr_for[tier],
                    "weight_decay": weight_decay if kind == "decay" else 0.0,
                }
            )
        return out

    def _encode_qwen_branch(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        visual: nn.Module,
        no_grad: bool,
        pool: bool = True,
    ) -> torch.Tensor:
        """Return Qwen3.5 post-merger tokens, optionally mean-pooled per sample."""
        pixel_values = pixel_values.to(dtype=next(visual.parameters()).dtype)
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            output = visual(pixel_values, grid_thw=image_grid_thw)
            embeds = output.pooler_output
        if not pool:
            return embeds
        return _split_and_pool(embeds, image_grid_thw, self.vit_merge_size**2)

    def encode_visual(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        frozen: bool = False,
        pool: bool = True,
    ) -> torch.Tensor:
        visual = self.frozen_visual if frozen else self.trainable_visual
        return self._encode_qwen_branch(pixel_values, grid_thw, visual=visual, no_grad=frozen, pool=pool)

    def _patchify_pseudo_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify ``(B,F,3,H,W)`` like Qwen3.5, padding an odd final frame."""
        p, m, tp = self.vit_patch_size, self.vit_merge_size, self.vit_temporal_patch_size
        batch, frames, _, height, width = video.shape
        if frames % tp:
            padding = video[:, -1:].expand(-1, tp - frames % tp, -1, -1, -1)
            video = torch.cat((video, padding), dim=1)
            frames = video.shape[1]

        grid_t, grid_h, grid_w = frames // tp, height // p, width // p
        patches = video.reshape(batch, grid_t, tp, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = patches.reshape(batch * grid_t * grid_h * grid_w, 3 * tp * p * p)
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch,
            device=video.device,
            dtype=torch.long,
        )
        return pixel_values, grid_thw

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

    @staticmethod
    def _normalize_scalar_rows(x: torch.Tensor, mode: str = "robust") -> torch.Tensor:
        """Map robust or already-standardized scalar rows to ``[-1, 1]``."""
        if mode == "robust":
            return MultimodalAlignmentModel._robust_normalize_rows(x)
        if mode == "prestandardized":
            return torch.nan_to_num(x.float()).clamp(-4.0, 4.0) / 4.0
        raise ValueError(f"unknown scalar normalization mode {mode!r}")

    @staticmethod
    def _normalize_tactile(x: torch.Tensor) -> torch.Tensor:
        """Apply OpenTouch's fixed pressure scale and map it to ``[-1, 1]``."""
        return torch.nan_to_num(x.float()).clamp(0.0, _TACTILE_PRESSURE_MAX).mul(
            2.0 / _TACTILE_PRESSURE_MAX
        ).sub(1.0)

    def _timeseries_to_video_inputs(
        self, signal: torch.Tensor, normalization: str = "robust"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack consecutive merger-cell-width signal blocks as video frames."""
        cell = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(signal)
        raw = torch.nan_to_num(signal.float())
        value = self._normalize_scalar_rows(raw, normalization)
        value = value.masked_fill(~finite, -1.0)

        channels, steps = value.shape
        frame_count = math.ceil(steps / cell)
        frames = value.new_full((frame_count, 3, channels * cell, cell), -1.0)
        for frame in range(frame_count):
            start, end = frame * cell, min((frame + 1) * cell, steps)
            length = end - start
            tile = value[:, start:end].repeat_interleave(cell, dim=0)
            frames[frame, :, :, :length] = tile.unsqueeze(0).expand(3, -1, -1)
        return self._patchify_pseudo_video(frames.unsqueeze(0))

    def _tactile_frame_tiles(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build ``(T,3,S,2S)`` tactile/force tiles at merger-cell resolution."""
        side = self.vit_patch_size * self.vit_merge_size
        tac = payload["tactile"]
        force = payload.get("force")
        finite = torch.isfinite(tac)
        value = self._normalize_tactile(tac)
        value = value.masked_fill(~finite, -1.0)
        value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest").squeeze(1)

        frame_count = value.shape[0]
        if self.tactile_delta_channels:
            delta = torch.zeros_like(value)
            delta[1:] = (value[1:] - value[:-1]) * 0.5
            tactile_frames = torch.stack((value, delta, value), dim=1)
        else:
            tactile_frames = value.unsqueeze(1).expand(-1, 3, -1, -1).clone()

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

    def _tactile_to_video_inputs(
        self,
        payload: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed tactile and force tiles through Qwen's temporal patch path."""
        frames = self._tactile_frame_tiles(payload)
        return self._patchify_pseudo_video(frames.unsqueeze(0))

    def encode_ts_trainable(
        self,
        signals: list[torch.Tensor | dict[str, torch.Tensor]],
        formats: list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Render mixed native-shape signals and encode them in one vision pass."""
        dev = device or next(self.trainable_visual.parameters()).device
        pvs, grids = [], []
        for sig, fmt in zip(signals, formats, strict=True):
            s = (
                {key: value.to(device=dev) for key, value in sig.items()}
                if isinstance(sig, dict)
                else sig.to(device=dev)
            )
            if fmt == "tactile":
                pv, g = self._tactile_to_video_inputs(s)
            else:
                normalization = self.ecg_normalization if fmt == "ecg" else "robust"
                pv, g = self._timeseries_to_video_inputs(s, normalization=normalization)
            pvs.append(pv)
            grids.append(g)
        pixel_values = torch.cat(pvs, dim=0)
        grid_thw = torch.cat(grids, dim=0)
        return self._encode_qwen_branch(pixel_values, grid_thw, visual=self.trainable_visual, no_grad=False)

    @torch.no_grad()
    def encode_text_chunks(
        self,
        texts: list[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode every non-overlapping SigLIP2-length chunk and its answer index."""
        max_length = max_length or int(self.label_text_model.config.max_position_embeddings)
        toks = self.label_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        owners = toks.pop("overflow_to_sample_mapping").to(device)
        chunks = self.label_text_model(**toks.to(device)).pooler_output.float()
        return chunks, owners

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        chunks, owners = self.encode_text_chunks(texts, device, max_length)
        pooled = chunks.new_zeros((len(texts), chunks.shape[-1]))
        pooled.index_add_(0, owners, chunks)
        counts = torch.bincount(owners, minlength=len(texts)).to(chunks.dtype)
        return pooled / counts.unsqueeze(1)

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """L2-normalize with a mixed-precision-safe epsilon."""
        return F.normalize(x, dim=-1, eps=1e-6) if x.numel() > 0 else x

    def project(self, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(head.parameters()).dtype)
        return self._norm(head(x))
