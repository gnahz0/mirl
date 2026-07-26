# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Multimodal alignment model wrapper.

Holds:
    - ``trainable_visual``  : the exact Qwen3.5 ``model.visual`` tower (trainable).
    - ``frozen_visual``     : an identical frozen copy (image-distillation teacher).
    - ``label_text_model``  : the frozen SigLIP2-SO400M text tower used to encode labels.
    - ``proj_visual`` / ``proj_text`` : heads into the shared contrastive dim (ts <-> text).
    - ``log_logit_scale``  : learnable contrastive temperature.

Losses (two):
    - ``ts_text``     : InfoNCE between proj(trainable VE on signal pixels) and
                        proj(SigLIP2 label text). This teaches the VE the new modality.
    - ``distill_img`` : cosine distance between normalized raw VE features
                        on images/videos -- no projection heads involved, so it directly
                        anchors the VE outputs the LM tower will consume in Stage 2.

Vision encoding details:
    Qwen3.5's ``model.visual(pixel_values, grid_thw=image_grid_thw)`` returns a
    ``BaseModelOutputWithPooling``. ``last_hidden_state`` contains the 1152-D
    pre-merger patch states, while ``pooler_output`` contains the 4096-D post-merger
    tokens that are injected into the language model. Alignment and distillation use
    ``pooler_output``.
    ``image_grid_thw`` rows are the processor's PRE-merge ``(t, h, w)``, so each
    sample's row count is ``t * h * w / spatial_merge_size**2``. We split on those
    counts and mean-pool per sample to ``[B, hidden]``.

Time-series encoding:
    Raw signals become merger-aligned pseudo-images or pseudo-videos in the exact
    flattened layout of Qwen3.5's processor. There is no line plot, PIL conversion,
    or separate time-series encoder.

TODO(stage2): export the trained ``trainable_visual.state_dict()`` back into a full
Qwen3.5 HF checkpoint and point veRL's ``actor_rollout_ref.model.path`` at it.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import ProjectionHead

logger = logging.getLogger(__name__)


def _resolve_snapshot(path_or_repo: str) -> Path:
    """Resolve a local HF snapshot, downloading it only when a repo id is given."""
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo)).resolve()


def _load_exact_qwen35_visual(
    path_or_repo: str,
    *,
    dtype: torch.dtype,
    attn_impl: str,
) -> nn.Module:
    """Load only the native Qwen3.5 ``model.visual`` tensors.

    Loading ``Qwen3_5ForConditionalGeneration`` just to retain ``model.visual`` would
    temporarily materialize the entire 9B model. This loader instantiates the same
    ``Qwen3_5VisionModel`` class and strictly loads the ``model.visual.*`` tensors from
    the Qwen3.5 checkpoint. Strict loading is deliberate: an architecture or checkpoint
    mismatch must fail instead of silently constructing a merely similar SigLIP tower.
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    root = _resolve_snapshot(path_or_repo)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    vision_config = full_config.vision_config
    vision_config._attn_implementation = attn_impl
    visual = Qwen3_5VisionModel(vision_config).to(dtype=dtype)

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted(
            {name for key, name in weight_map.items() if key.startswith("model.visual.")}
        )
    elif (root / "model.safetensors").exists():
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint found under {root}")

    state_dict: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.visual."):
                    state_dict[key.removeprefix("model.visual.")] = handle.get_tensor(key)
                elif key.startswith("visual."):
                    state_dict[key.removeprefix("visual.")] = handle.get_tensor(key)

    if not state_dict:
        raise RuntimeError(f"no model.visual tensors found in {root}")
    visual.load_state_dict(state_dict, strict=True)
    return visual


def _load_exact_siglip2_text(path_or_repo: str, *, dtype: torch.dtype) -> nn.Module:
    """Load only ``text_model.*`` from a paired SigLIP2 checkpoint, strictly."""
    from safetensors import safe_open
    from transformers import AutoConfig, Siglip2TextModel

    root = _resolve_snapshot(path_or_repo)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    text_model = Siglip2TextModel(full_config.text_config).to(dtype=dtype)

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted({name for key, name in weight_map.items() if key.startswith("text_model.")})
    elif (root / "model.safetensors").exists():
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors checkpoint found under {root}")

    state_dict: dict[str, torch.Tensor] = {}
    for filename in filenames:
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("text_model."):
                    state_dict[key] = handle.get_tensor(key)
    if not state_dict:
        raise RuntimeError(f"no text_model tensors found in {root}")
    text_model.load_state_dict(state_dict, strict=True)
    return text_model


def _enable_block_checkpointing(visual: nn.Module) -> int:
    """Wrap each vision transformer block's forward in non-reentrant activation
    checkpointing and return the number of blocks wrapped.

    Qwen3.5's vision ``forward`` loops over ``self.blocks`` directly and never
    consults ``self.gradient_checkpointing``, so HF's built-in
    ``gradient_checkpointing_enable()`` is a no-op for this tower. We patch the
    bound ``forward`` on each block instead -- this adds NO submodules, so every
    parameter name is preserved and existing checkpoints stay loadable. The
    wrapper only checkpoints while the block is in training mode (auto-disabled
    during eval/validation, where there is no backward pass to trade against).
    """
    import torch.utils.checkpoint as cp

    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        return 0
    n = 0
    for blk in blocks:
        if getattr(blk, "_ckpt_wrapped", False):
            continue
        orig_forward = blk.forward

        def make(orig, module):
            def fwd(*args, **kwargs):
                if module.training:
                    return cp.checkpoint(orig, *args, use_reentrant=False, **kwargs)
                return orig(*args, **kwargs)
            return fwd

        blk.forward = make(orig_forward, blk)
        blk._ckpt_wrapped = True
        n += 1
    return n


def _split_and_pool(
    flat_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_unit: int,
) -> torch.Tensor:
    """flat_embeds: [sum_i n_i / merge_unit, hidden] -- the VE output is POST-merger,
    so each sample contributes ``t*h*w / merge_unit`` rows (``grid_thw`` rows are the
    processor's PRE-merge ``(t, h, w)``).

    Returns: [B, hidden] mean-pooled per sample.
    """
    if flat_embeds.numel() == 0 or grid_thw is None or grid_thw.numel() == 0:
        return flat_embeds.new_zeros((0, flat_embeds.shape[-1] if flat_embeds.ndim >= 1 else 1))
    counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2] // merge_unit).tolist()
    if sum(counts) != flat_embeds.shape[0]:
        raise ValueError(
            f"VE output rows ({flat_embeds.shape[0]}) != sum of per-sample post-merge "
            f"counts ({sum(counts)}); grid_thw={grid_thw.tolist()}, merge_unit={merge_unit}"
        )
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
        qwen35_path: HF id or local path for the full Qwen3.5 checkpoint.
        siglip2_text_path: HF id or local path for the paired SigLIP2-SO400M
            checkpoint. Only its text tower is retained.
        shared_dim: projection output dim.
        proj_hidden_dim: projection MLP hidden width.
        visual_dtype: dtype for both VEs (bf16 strongly recommended).
        attn_impl: attention implementation used by the Qwen3.5 vision towers.
        ts_representation: ``image`` for all merger-aware pseudo-images, ``video``
            for all pseudo-videos, or ``hybrid`` (direct-value pseudo-images for
            SmellNet/ECG and native pseudo-video for spatial tactile frames).
    """

    def __init__(
        self,
        qwen35_path: str = "Qwen/Qwen3.5-9B",
        siglip2_text_path: str = "google/siglip2-so400m-patch16-naflex",
        shared_dim: int = 512,
        proj_hidden_dim: Optional[int] = 1024,
        proj_dropout: float = 0.0,
        visual_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        gradient_checkpointing: bool = False,
        ts_representation: str = "hybrid",
    ):
        super().__init__()
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
        )

        import time as _time

        # ---- Qwen3.5 processor (shared between trainable & frozen VE) ----
        logger.info("[1/4] loading Qwen3.5 processor from %s", qwen35_path)
        _t = _time.time()
        qwen_root = _resolve_snapshot(qwen35_path)
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_root, local_files_only=True)
        logger.info("       processor ready (%.1fs)", _time.time() - _t)

        # ---- Trainable VE: exact native model.visual class and checkpoint tensors ----
        logger.info("[2/4] loading exact Qwen3.5 model.visual weights (dtype=%s, attn=%s)",
                    visual_dtype, attn_impl)
        _t = _time.time()
        self.trainable_visual = _load_exact_qwen35_visual(
            str(qwen_root), dtype=visual_dtype, attn_impl=attn_impl,
        )
        logger.info("       trainable VE ready: %.1fM params (%.1fs)",
                    sum(p.numel() for p in self.trainable_visual.parameters()) / 1e6,
                    _time.time() - _t)

        # ---- Frozen reference VE (right): identical weights, requires_grad=False ----
        logger.info("[3/4] cloning frozen reference vision encoder (deepcopy on CPU)")
        _t = _time.time()
        self.frozen_visual = copy.deepcopy(self.trainable_visual)
        for p in self.frozen_visual.parameters():
            p.requires_grad_(False)
        self.frozen_visual.eval()
        logger.info("       frozen VE ready (%.1fs)", _time.time() - _t)

        # Gradient checkpointing on the TRAINABLE VE only (frozen runs under no_grad,
        # so there is nothing to checkpoint). Applied AFTER the frozen deepcopy so the
        # frozen tower keeps its clean, un-patched forward.
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            n_ckpt = _enable_block_checkpointing(self.trainable_visual)
            logger.info("       gradient checkpointing ON: wrapped %d trainable VE blocks", n_ckpt)

        qwen_hidden = self._infer_qwen_visual_hidden(self.trainable_visual)

        # Patch geometry needed to format raw signals exactly like image inputs
        # (signal channels -> patch rows, time -> patch columns).
        vcfg = getattr(self.trainable_visual, "config", None)
        self.vit_patch_size = int(getattr(vcfg, "patch_size", 16))
        self.vit_merge_size = int(getattr(vcfg, "spatial_merge_size", 2))
        self.vit_temporal_patch_size = int(getattr(vcfg, "temporal_patch_size", 2))
        logger.info(
            "[ts] signal-as-image formatting: patch_size=%d merge_size=%d temporal_patch_size=%d",
            self.vit_patch_size, self.vit_merge_size, self.vit_temporal_patch_size,
        )

        # ---- SigLIP2 label-text encoder (frozen) ----
        #
        # Qwen3.5's native visual tower has the SigLIP2-SO400M architecture and was
        # initialized from those weights, but Qwen subsequently VL-trained it. The
        # original paired SigLIP2 text tower is therefore the closest contrastive text
        # teacher, not an assertion that its final space is identical to Qwen3.5's.
        # Learned projection heads bridge that expected drift.
        logger.info("[4/4] loading SigLIP2 label-text encoder %s", siglip2_text_path)
        _t = _time.time()
        siglip_root = _resolve_snapshot(siglip2_text_path)
        self.label_tokenizer = AutoTokenizer.from_pretrained(siglip_root, local_files_only=True)
        self.label_text_model = _load_exact_siglip2_text(
            str(siglip_root), dtype=visual_dtype
        )
        for p in self.label_text_model.parameters():
            p.requires_grad_(False)
        self.label_text_model.eval()
        label_hidden = self.label_text_model.config.projection_size
        logger.info(
            "       SigLIP2 text ready: hidden=%d (%.1fs)",
            label_hidden,
            _time.time() - _t,
        )

        # ---- Projection heads (2 total; contrastive space only) ----
        # The ts contrastive loss compares the trainable VE output (at ``qwen_hidden``)
        # with SigLIP2 text. Distillation happens on RAW VE features (no projection), so
        # no head is needed for the frozen reference VE.
        self.proj_visual = ProjectionHead(qwen_hidden, shared_dim, proj_hidden_dim, proj_dropout)
        self.proj_text = ProjectionHead(label_hidden, shared_dim, proj_hidden_dim, proj_dropout)

        # ---- Learnable contrastive temperature ----
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        self.shared_dim = shared_dim
        self.qwen_hidden = qwen_hidden
        self.label_hidden = label_hidden
        self.visual_dtype = visual_dtype
        if ts_representation not in {"image", "video", "hybrid"}:
            raise ValueError(
                "ts_representation must be one of 'image', 'video', or 'hybrid', "
                f"got {ts_representation!r}"
            )
        self.ts_representation = ts_representation

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_qwen_visual_hidden(visual_module: nn.Module) -> int:
        """The Qwen3.5 visual tower exposes its output dim via config.out_hidden_size
        (post merger). Fall back to inspecting the merger head if needed."""
        cfg = getattr(visual_module, "config", None)
        if cfg is not None:
            for attr in ("out_hidden_size", "hidden_size"):
                if hasattr(cfg, attr):
                    return int(getattr(cfg, attr))
        for name, mod in visual_module.named_modules():
            if isinstance(mod, nn.Linear) and "merger" in name:
                return mod.out_features
        raise RuntimeError("could not infer Qwen3.5 visual output dim")

    def trainable_parameter_groups(self, lr: float, weight_decay: float, head_lr: Optional[float] = None):
        """Param groups for the optimizer. Frozen VE and label text encoder are excluded.

        Two LR tiers:
          * the pretrained Qwen3.5 ViT (``trainable_visual.*``) trains at ``lr``;
          * the from-scratch modules (proj_visual/text, logit_scale) train at
            ``head_lr`` (defaults to ``lr`` if not given).
        Bias / LayerNorm / Norm params get weight_decay=0 (standard practice).
        """
        head_lr = lr if head_lr is None else head_lr
        groups = {
            ("vit", "decay"): [], ("vit", "no_decay"): [],
            ("head", "decay"): [], ("head", "no_decay"): [],
        }
        for p_name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            tier = "vit" if p_name.startswith("trainable_visual.") else "head"
            no_decay = p.ndim == 1 or p_name.endswith(".bias") or "logit_scale" in p_name
            groups[(tier, "no_decay" if no_decay else "decay")].append(p)
        lr_for = {"vit": lr, "head": head_lr}
        out = []
        for (tier, kind), params in groups.items():
            if not params:
                continue
            out.append({
                "params": params,
                "lr": lr_for[tier],
                "weight_decay": weight_decay if kind == "decay" else 0.0,
            })
        return out

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
        """Return ``[B, 4096]`` means of Qwen3.5's post-merger visual tokens."""
        if pixel_values.numel() == 0 or image_grid_thw.numel() == 0:
            return pixel_values.new_zeros((0, self.qwen_hidden))
        pixel_values = pixel_values.to(dtype=visual.dtype if hasattr(visual, "dtype") else self.visual_dtype)
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            output = visual(pixel_values, grid_thw=image_grid_thw)
            embeds = output.pooler_output
        return _split_and_pool(embeds, image_grid_thw, self.vit_merge_size ** 2)

    def encode_images_trainable(self, pixel_values, image_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values, image_grid_thw, visual=self.trainable_visual, no_grad=False
        )

    def encode_images_frozen(self, pixel_values, image_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values, image_grid_thw, visual=self.frozen_visual, no_grad=True
        )

    # Videos use the same Qwen3.5 visual tower; the only thing that changes is
    # ``grid_thw[:, 0] > 1`` (multiple frames per clip). We keep these as separate
    # methods so the trainer reads more naturally and we have a hook point if we
    # ever want video-specific pooling.
    def encode_videos_trainable(self, pixel_values_videos, video_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values_videos, video_grid_thw, visual=self.trainable_visual, no_grad=False
        )

    def encode_videos_frozen(self, pixel_values_videos, video_grid_thw) -> torch.Tensor:
        return self._encode_qwen_branch(
            pixel_values_videos, video_grid_thw, visual=self.frozen_visual, no_grad=True
        )

    # ---- Time-series path: merger-aware pseudo-image / pseudo-video -----------------

    def _patchify_pseudo_image(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Carve a pseudo-image ``(B, 3, H, W)`` into Qwen patches.

        ``H`` and ``W`` must already be divisible by ``patch_size * merge_size``.
        Reproduces the exact flatten layout of Qwen3.5's image processor
        (temporal replicate, split H/W into ``(grid//merge, merge, patch)``, flatten
        per patch) so signals and real images share the ``(pixel_values, grid_thw)``
        interface.

        Returns:
            pixel_values: ``[B * grid_h * grid_w, 3 * temporal_patch_size * patch_size**2]``
            grid_thw:     ``[B, 3]`` rows of ``(1, grid_h=H/patch, grid_w=W/patch)``
        """
        tp = self.vit_temporal_patch_size
        video = img.unsqueeze(1).expand(-1, tp, -1, -1, -1)
        return self._patchify_pseudo_video(video)

    def _patchify_pseudo_video(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify ``(B,F,3,H,W)`` exactly like Qwen3.5's video processor.

        Consecutive pairs are fused by the native ``temporal_patch_size=2`` Conv3d
        kernel before the spatial 2x2 merger. An odd final frame is repeated, matching
        the processor's temporal padding behavior.
        """
        p, m, tp = self.vit_patch_size, self.vit_merge_size, self.vit_temporal_patch_size
        b, frames, channels, H, W = video.shape
        if channels != 3 or frames < 1:
            raise ValueError(f"expected non-empty (B,F,3,H,W) video, got {tuple(video.shape)}")
        if H % (p * m) or W % (p * m):
            raise ValueError(f"video H/W must be divisible by {p * m}, got {(H, W)}")
        if frames % tp:
            video = torch.cat((video, video[:, -1:].expand(-1, tp - frames % tp, -1, -1, -1)), dim=1)
            frames = video.shape[1]

        grid_t, grid_h, grid_w = frames // tp, H // p, W // p
        patches = video.reshape(
            b, grid_t, tp, 3, grid_h // m, m, p, grid_w // m, m, p
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = patches.reshape(b * grid_t * grid_h * grid_w, 3 * tp * p * p)
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * b, device=video.device, dtype=torch.long
        )
        return pixel_values, grid_thw

    @staticmethod
    def _robust_normalize_rows(x: torch.Tensor) -> torch.Tensor:
        """TimeOmni-style robust fidelity normalization, independently per row.

        The median/MAD component resists sparse pressure spikes and sensor outliers,
        while the standard-deviation component still represents broad variation.
        ``tanh`` maps directly to the ``[-1, 1]`` range Qwen's image normalization
        normally produces, without a PIL/uint8 quantization round trip.
        """
        x = torch.nan_to_num(x.float())
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values / 0.6745
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        scale = (0.5 * mad + 0.5 * std).clamp_min(1e-6)
        return torch.tanh(centered / (4.0 * scale))

    @staticmethod
    def _pad_to(value: int, unit: int) -> int:
        return ((int(value) + unit - 1) // unit) * unit

    def _timeseries_to_pixel_inputs(
        self, signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one unified direct-value raster to any scalar-channel ``(C,T)`` series.

        Consecutive samples snake through one merger-cell-high band per channel.
        Every timestep occupies exactly one pixel (no plotting, interpolation,
        downsampling, or modality-specific period estimation), neighboring columns
        reverse direction to keep the 1-D path spatially continuous, and semantic
        channels never share a post-merger token.

        A 32x32 merger cell carries at most 1024 timesteps from one channel. Longer
        sequences grow horizontally in complete 32px blocks. Thus native SmellNet and
        ECG use exactly the same normalization, grayscale intensity, packing rule, padding, and
        Qwen patch/merger contract; only ``C`` and ``T`` differ.
        """
        cell = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(signal)
        raw = torch.nan_to_num(signal.float())
        value = self._robust_normalize_rows(raw)
        value = value.masked_fill(~finite, -1.0)

        channels, steps = value.shape
        packed_cols = math.ceil(steps / cell)
        width = self._pad_to(packed_cols, cell)
        # Exact -1 is the padding/missing sentinel. The same scalar intensity is
        # repeated over RGB so Qwen sees a grayscale numerical image rather than
        # three unrelated engineered feature planes.
        img = value.new_full((3, channels * cell, width), -1.0)

        time_idx = torch.arange(steps, device=value.device)
        cols = time_idx // cell
        phase = time_idx % cell
        rows = torch.where(cols.remainder(2) == 0, phase, cell - 1 - phase)
        for channel in range(channels):
            raster_rows = channel * cell + rows
            img[:, raster_rows, cols] = value[channel].unsqueeze(0).expand(3, -1)
        return self._patchify_pseudo_image(img.unsqueeze(0))

    def _smell_to_pixel_inputs(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._timeseries_to_pixel_inputs(signal)

    def _timeseries_to_video_inputs(
        self, signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use each fixed 32-step scalar-series window as one pseudo-video frame.

        A temporal patch fuses two adjacent windows, then the 2x2 merger emits one
        token per channel. This is a frequency-free all-video ablation shared by ECG
        and SmellNet; the default hybrid path uses the denser scalar pseudo-image.
        """
        cell = self.vit_patch_size * self.vit_merge_size
        finite = torch.isfinite(signal)
        raw = torch.nan_to_num(signal.float())
        value = self._robust_normalize_rows(raw)
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

    def _ecg_to_pixel_inputs(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._timeseries_to_pixel_inputs(signal)

    def _smell_to_video_inputs(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._timeseries_to_video_inputs(signal)

    def _ecg_to_video_inputs(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._timeseries_to_video_inputs(signal)

    def _tactile_frame_tiles(
        self, payload: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Build ``(T,3,32,64)`` tactile+right-force frame tiles."""
        side = self.vit_patch_size * self.vit_merge_size
        if isinstance(payload, dict):
            tac = payload["tactile"]
            force = payload.get("force")
        else:
            tac = payload
            force = None
        finite = torch.isfinite(tac)
        raw = torch.nan_to_num(tac.float())
        # Tactile taxels share a physical unit and form one pressure surface. A single
        # recording-level scale preserves relative pressure across the 16x16 contact
        # map; normalizing each taxel independently would erase that spatial signal.
        value = self._robust_normalize_rows(raw.reshape(1, -1)).reshape_as(raw)
        value = value.masked_fill(~finite, -1.0)
        value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest").squeeze(1)

        frame_count = value.shape[0]
        tactile_frames = value.unsqueeze(1).expand(-1, 3, -1, -1).clone()

        # The adjacent 32x32 cell carries the 13 right-hand force summaries. v2's
        # left-hand columns were filtered by the loader so its schema matches v1.
        force_frames = value.new_full((frame_count, 3, side, side), -1.0)
        if force is not None and force.numel() > 0:
            force = force.float()
            if force.shape[0] != frame_count:
                raise ValueError(
                    f"tactile/force frame mismatch: {frame_count} vs {force.shape[0]}"
                )
            force_finite = torch.isfinite(force)
            force_raw = torch.nan_to_num(force)
            force_value = self._robust_normalize_rows(force_raw.t()).t()
            force_value = force_value.masked_fill(~force_finite, -1.0)
            num_force = force.shape[1]
            for channel in range(num_force):
                row_start = channel * side // num_force
                row_end = (channel + 1) * side // num_force
                encoded = force_value[:, channel, None, None, None]
                force_frames[:, :, row_start:row_end] = encoded.expand(
                    -1, 3, row_end - row_start, side
                )

        return torch.cat((tactile_frames, force_frames), dim=-1)

    def _tactile_to_pixel_inputs(
        self, payload: dict[str, torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map tactile+force frame tiles to a serpentine contact sheet."""
        side = self.vit_patch_size * self.vit_merge_size
        frame_tiles = self._tactile_frame_tiles(payload)
        frame_count = frame_tiles.shape[0]
        cols = max(1, math.ceil(math.sqrt(frame_count / 2)))
        rows = math.ceil(frame_count / cols)
        img = frame_tiles.new_full((3, rows * side, cols * side * 2), -1.0)
        for frame in range(frame_count):
            row = frame // cols
            offset = frame % cols
            col = offset if row % 2 == 0 else cols - 1 - offset
            rs, cs = row * side, col * side * 2
            img[:, rs : rs + side, cs : cs + side * 2] = frame_tiles[frame]
        return self._patchify_pseudo_image(img.unsqueeze(0))

    def _tactile_to_video_inputs(
        self, payload: dict[str, torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed tactile+force tiles through Qwen3.5's native temporal patch path.

        Each source frame contributes two spatial merger cells (right tactile map and
        right-force summary). The temporal kernel combines adjacent frame pairs, yielding
        two 4096-D tokens per pair.
        """
        frames = self._tactile_frame_tiles(payload)
        return self._patchify_pseudo_video(frames.unsqueeze(0))

    def encode_ts_trainable(
        self,
        signals: list[torch.Tensor | dict[str, torch.Tensor]],
        formats: list[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Encode a list of native-shape signals through the trainable VE (with grad).

        ``signals`` and ``formats`` are parallel lists from the collator. Each sample is
        formatted to its own ``(pixel_values, grid_thw)`` (the same direct-value
        channel raster for ``smell``/``ecg``, spatial frames for ``tactile``), then
        all are concatenated so the VE runs once and the batch can mix native shapes.
        Returns ``[N, qwen_hidden]`` (or ``None``).
        """
        if not signals:
            return None
        dev = device or next(self.trainable_visual.parameters()).device
        pvs, grids = [], []
        for sig, fmt in zip(signals, formats):
            s = (
                {key: value.to(device=dev) for key, value in sig.items()}
                if isinstance(sig, dict)
                else sig.to(device=dev)
            )
            use_video = self.ts_representation == "video" or (
                self.ts_representation == "hybrid" and fmt == "tactile"
            )
            if fmt == "tactile":
                pv, g = (
                    self._tactile_to_video_inputs(s)
                    if use_video
                    else self._tactile_to_pixel_inputs(s)
                )
            elif fmt == "ecg":
                pv, g = self._ecg_to_video_inputs(s) if use_video else self._ecg_to_pixel_inputs(s)
            elif fmt == "smell":
                pv, g = (
                    self._smell_to_video_inputs(s)
                    if use_video
                    else self._smell_to_pixel_inputs(s)
                )
            else:
                raise ValueError(f"unknown time-series format {fmt!r}")
            pvs.append(pv)
            grids.append(g)
        pixel_values = torch.cat(pvs, dim=0)
        grid_thw = torch.cat(grids, dim=0)
        return self._encode_qwen_branch(
            pixel_values, grid_thw, visual=self.trainable_visual, no_grad=False
        )

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.label_hidden), device=device)
        max_length = max_length or int(self.label_text_model.config.max_position_embeddings)
        # SigLIP2 was trained with fixed-length padding, and its pooler reads the final
        # token. Using padding="max_length" is therefore part of the model contract.
        toks = self.label_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = self.label_text_model(**toks)
        return out.pooler_output

    # -------------------------------------------------------------------------
    # Projection + normalize convenience
    # -------------------------------------------------------------------------

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """L2-normalize with a *generous* epsilon floor.

        Default ``F.normalize`` uses eps=1e-12 which is fine for fp32 but blows up
        in mixed precision: low-magnitude feature vectors can collapse to ||x|| << 1e-6
        after the visual encoder, giving 1e-6 / 1e-12 = 1e6-scale vectors that overflow
        downstream. eps=1e-6 keeps gradients well-conditioned without changing
        the unit-norm property for any feature with reasonable magnitude.
        """
        return F.normalize(x, dim=-1, eps=1e-6) if x.numel() > 0 else x

    def project(self, head: ProjectionHead, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, self.shared_dim))
        # Cast to fp32 explicitly so the entire projection runs in full precision
        # regardless of the visual encoder's dtype (avoids bf16 underflow in MLP).
        x = x.to(next(head.parameters()).dtype)
        return self._norm(head(x))
