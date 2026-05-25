# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Single-GPU Stage 1 alignment trainer. No Ray, no veRL, no FSDP.

Usage
-----
    python -m verl.trainer.alignment.trainer \\
        --config verl/trainer/alignment/config/stage1_qwen3vl_clip.yaml

The config is a small dict-of-dicts loaded with PyYAML or OmegaConf (we use OmegaConf
since the rest of the repo already depends on it).

What this does
--------------
For each batch of image-bearing samples from the existing JSONL:

    1. Split by ``data_source`` into ``img`` and ``ts`` branches.
    2. Run each branch's PIL list through the Qwen3-VL processor -> pixel_values + image_grid_thw.
    3. Encode through trainable Qwen3-VL ``.visual`` (with grad).
    4. Encode through the frozen reference Qwen3-VL ``.visual`` (no grad).
    5. Encode the per-sample text labels through frozen CLIP text encoder.
    6. Project + L2 normalize all five embedding sets to a shared dim.
    7. Compute the configured contrastive + distillation losses, sum with weights, backprop.

TODOs marked inline for the Stage 2 / veRL handoff.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data import AlignmentDataset, collate_alignment
from .losses import distill_kl_on_sim, distill_mse, info_nce_symmetric
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "float32": torch.float32,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16}[name.lower()]


def _maybe_init_wandb(cfg: DictConfig):
    """Initialize a W&B run if enabled in config. Returns the run handle or ``None``.

    Robust to:
        * ``wandb`` not installed -> warn and continue without it.
        * No API key on the machine -> falls back to ``WANDB_MODE=offline``.
        * ``cfg.wandb.mode`` explicitly set to ``"disabled"`` / ``"offline"`` / ``"online"``.
    """
    wcfg = cfg.get("wandb", {}) or {}
    if not wcfg.get("enable", False):
        logger.info("W&B disabled in config; skipping init")
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; continuing without it (`pip install wandb` to enable)")
        return None

    # If user set WANDB_API_KEY env, online mode works. Otherwise fall back to offline.
    if wcfg.get("mode"):
        os.environ.setdefault("WANDB_MODE", str(wcfg.get("mode")))
    elif "WANDB_API_KEY" not in os.environ and os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
        logger.warning(
            "WANDB_API_KEY not set; defaulting to WANDB_MODE=offline. "
            "Run `wandb login` (or set WANDB_API_KEY) to log online."
        )
        os.environ["WANDB_MODE"] = "offline"

    try:
        run = wandb.init(
            project=str(wcfg.get("project", "mirl-alignment")),
            name=str(wcfg.get("name", "stage1")),
            entity=str(wcfg.get("entity")) if wcfg.get("entity") else None,
            tags=list(wcfg.get("tags", []) or []),
            notes=str(wcfg.get("notes", "")) if wcfg.get("notes") else None,
            group=str(wcfg.get("group")) if wcfg.get("group") else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("W&B run initialized: %s (mode=%s)",
                    run.url if hasattr(run, "url") else "?", os.environ.get("WANDB_MODE", "online"))
        return run
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb init failed (%s); continuing without it", e)
        return None


def _process_images(processor, pil_list, device, dtype):
    """Run the Qwen3-VL processor on a list of PIL images."""
    if not pil_list:
        return None
    out = processor(
        images=pil_list,
        text=["<image>"] * len(pil_list),
        return_tensors="pt",
        padding=True,
    )
    return {
        "pixel_values": out["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw": out["image_grid_thw"].to(device=device),
    }


def _process_videos(processor, video_list, device, dtype):
    """Run the Qwen3-VL processor on a list of ``(video_tensor, video_metadata)`` pairs.

    ``video_list`` matches the format returned by ``vision_utils.process_video(...,
    return_video_metadata=True)`` -- i.e. each entry is ``(tensor[n_frames, 3, H, W], meta)``.
    """
    if not video_list:
        return None
    tensors, metas = zip(*video_list, strict=True)
    out = processor(
        videos=list(tensors),
        text=["<video>"] * len(tensors),
        return_tensors="pt",
        padding=True,
        videos_kwargs={"video_metadata": list(metas), "do_sample_frames": False},
    )
    return {
        "pixel_values_videos": out["pixel_values_videos"].to(device=device, dtype=dtype),
        "video_grid_thw": out["video_grid_thw"].to(device=device),
    }


def _encode_branch(
    model: MultimodalAlignmentModel,
    processor,
    images_pil,
    videos,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Encode one branch (img or ts) that may contain both images and videos.

    Returns ``(z_trainable, z_frozen)`` with shape ``[B_total, qwen_hidden]`` where rows are
    ordered ``[image_rows..., video_rows...]``. Either side may be ``None`` if the branch
    is empty.
    """
    img_in = _process_images(processor, images_pil, device, dtype)
    vid_in = _process_videos(processor, videos, device, dtype)
    if img_in is None and vid_in is None:
        return None, None

    train_parts: list[torch.Tensor] = []
    frozen_parts: list[torch.Tensor] = []

    if img_in is not None:
        t = model.encode_images_trainable(img_in["pixel_values"], img_in["image_grid_thw"])
        f = model.encode_images_frozen(img_in["pixel_values"], img_in["image_grid_thw"])
        train_parts.append(t)
        frozen_parts.append(f)
    if vid_in is not None:
        t = model.encode_videos_trainable(vid_in["pixel_values_videos"], vid_in["video_grid_thw"])
        f = model.encode_videos_frozen(vid_in["pixel_values_videos"], vid_in["video_grid_thw"])
        train_parts.append(t)
        frozen_parts.append(f)

    return torch.cat(train_parts, dim=0), torch.cat(frozen_parts, dim=0)


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    metrics: dict[str, float] = {}

    # Branch encodings (images + videos concatenated, in that order).
    feat_img, feat_ref_img = _encode_branch(
        model, model.qwen_processor,
        batch["img_image_pil"], batch["img_video"], device, visual_dtype,
    )
    feat_ts, feat_ref_ts = _encode_branch(
        model, model.qwen_processor,
        batch["ts_image_pil"], batch["ts_video"], device, visual_dtype,
    )

    img_text = list(batch["img_image_text"]) + list(batch["img_video_text"])
    ts_text = list(batch["ts_image_text"]) + list(batch["ts_video_text"])

    z_img = model.project(model.proj_img, feat_img) if feat_img is not None else None
    z_ref_img = model.project(model.proj_ref, feat_ref_img) if feat_ref_img is not None else None
    z_ts = model.project(model.proj_ts_img, feat_ts) if feat_ts is not None else None
    z_ref_ts = model.project(model.proj_ref_ts, feat_ref_ts) if feat_ref_ts is not None else None

    z_text_img = z_text_ts = None
    if z_img is not None and img_text:
        z_text_img = model.project(model.proj_text, model.encode_text(img_text, device=device).float())
    if z_ts is not None and ts_text:
        z_text_ts = model.project(model.proj_text, model.encode_text(ts_text, device=device).float())

    total = torch.zeros((), device=device, dtype=torch.float32)
    w = cfg.loss_weights

    if z_img is not None and z_text_img is not None and z_img.shape[0] > 1:
        l = info_nce_symmetric(z_img, z_text_img, model.log_logit_scale)
        total = total + float(w.img_text) * l
        metrics["loss/img_text"] = l.detach().item()

    if z_ts is not None and z_text_ts is not None and z_ts.shape[0] > 1:
        l = info_nce_symmetric(z_ts, z_text_ts, model.log_logit_scale)
        total = total + float(w.ts_text) * l
        metrics["loss/ts_text"] = l.detach().item()

    # Cross-modal image<->ts only makes sense if both branches happen to share a notion of
    # paired samples; for the default JSONL they don't (different rows -> different samples),
    # so we leave this off unless the user explicitly turns it on.
    if (
        float(w.img_ts) > 0.0
        and z_img is not None and z_ts is not None
        and z_img.shape[0] == z_ts.shape[0] and z_img.shape[0] > 1
    ):
        l = info_nce_symmetric(z_img, z_ts, model.log_logit_scale)
        total = total + float(w.img_ts) * l
        metrics["loss/img_ts"] = l.detach().item()

    if z_img is not None and z_ref_img is not None and z_img.shape[0] > 0:
        l = distill_mse(z_img, z_ref_img)
        total = total + float(w.distill_img) * l
        metrics["loss/distill_img"] = l.detach().item()

    if z_ts is not None and z_ref_ts is not None and z_ts.shape[0] > 0:
        l = distill_mse(z_ts, z_ref_ts)
        total = total + float(w.distill_ts) * l
        metrics["loss/distill_ts"] = l.detach().item()

    if float(getattr(w, "distill_kl", 0.0)) > 0.0:
        kls = []
        if z_img is not None and z_ref_img is not None and z_img.shape[0] >= 2:
            kls.append(distill_kl_on_sim(z_img, z_ref_img))
        if z_ts is not None and z_ref_ts is not None and z_ts.shape[0] >= 2:
            kls.append(distill_kl_on_sim(z_ts, z_ref_ts))
        if kls:
            kl = torch.stack(kls).mean()
            total = total + float(w.distill_kl) * kl
            metrics["loss/distill_kl"] = kl.detach().item()

    metrics["loss/total"] = total.detach().item()
    metrics["logit_scale"] = model.log_logit_scale.detach().exp().item()
    return total, metrics


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _resolve_dtype(cfg.train.amp_dtype)
    visual_dtype = _resolve_dtype(cfg.model.trainable_visual_dtype)

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir)

    # ---- Data ----
    train_ds = AlignmentDataset(
        data_files=list(cfg.data.train_files),
        ts_data_sources=list(cfg.data.ts_data_sources),
        text_for_clip=cfg.data.text_for_clip,
        max_samples=int(cfg.data.get("max_train_samples", -1)),
        balanced_sampling_key=cfg.data.get("balanced_sampling_key"),
        seed=int(cfg.train.get("seed", 42)),
        enable_videos=bool(cfg.data.get("enable_videos", True)),
        max_video_frames=cfg.data.get("max_video_frames"),
        image_patch_size=int(cfg.data.get("image_patch_size", 14)),
    )
    logger.info("train dataset: %d image-bearing samples", len(train_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.get("num_workers", 2)),
        collate_fn=collate_alignment,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model ----
    model = MultimodalAlignmentModel(
        qwen3_vl_path=str(cfg.model.qwen3_vl_path),
        clip_text_path=str(cfg.model.clip_text_path),
        shared_dim=int(cfg.projection.shared_dim),
        proj_hidden_dim=int(cfg.projection.hidden_dim),
        proj_dropout=float(cfg.projection.get("dropout", 0.0)),
        visual_dtype=visual_dtype,
        attn_impl=str(cfg.model.get("attn_impl", "sdpa")),
    ).to(device)

    # Ensure projection heads + log_logit_scale live in fp32 for stable optimization
    for p_name, p in model.named_parameters():
        if "proj_" in p_name or "log_logit_scale" in p_name:
            p.data = p.data.float()

    # ---- Optim ----
    optim_groups = model.trainable_parameter_groups(
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    optimizer = torch.optim.AdamW(
        optim_groups, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        betas=(0.9, 0.95), eps=1e-8,
    )

    total_steps = int(cfg.train.total_steps)
    warmup = int(cfg.train.warmup_steps)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        import math as _m
        return 0.5 * (1.0 + _m.cos(_m.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- Optional W&B ----
    wandb_run = _maybe_init_wandb(cfg)

    # ---- Train ----
    model.train()
    model.frozen_visual.eval()
    model.clip_text_model.eval()

    step = 0
    grad_accum = max(1, int(cfg.train.get("grad_accum_steps", 1)))
    log_every = int(cfg.train.get("log_every", 10))
    ckpt_every = int(cfg.train.get("ckpt_every", 500))

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    done = False
    while not done:
        for batch in train_loader:
            # Skip empty batches (rare, can happen if every sample failed to load).
            n_img = len(batch["img_image_pil"]) + len(batch["img_video"])
            n_ts = len(batch["ts_image_pil"]) + len(batch["ts_video"])
            if n_img == 0 and n_ts == 0:
                continue

            with autocast_ctx:
                loss, metrics = _compute_losses(model, batch, cfg, device, visual_dtype)

            if not torch.isfinite(loss):
                logger.warning("non-finite loss at step %d (%s); skipping", step, loss.item())
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if float(cfg.train.get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(cfg.train.grad_clip),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                counts = {
                    "n/img_image": len(batch["img_image_pil"]),
                    "n/img_video": len(batch["img_video"]),
                    "n/ts_image": len(batch["ts_image_pil"]),
                    "n/ts_video": len(batch["ts_video"]),
                }
                msg = (
                    f"step {step:6d} | lr {lr_now:.2e} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    + " | "
                    + " ".join(f"{k}={v}" for k, v in counts.items())
                    + f" | {elapsed:.1f}s"
                )
                logger.info(msg)
                if wandb_run is not None:
                    wandb_run.log({**metrics, **counts, "lr": lr_now, "step": step})

            if step > 0 and step % ckpt_every == 0:
                _save_checkpoint(model, out_dir / f"step_{step}", cfg)

            step += 1
            if step >= total_steps:
                done = True
                break

    _save_checkpoint(model, out_dir / "final", cfg)
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Stage 1 alignment complete. Saved to %s", out_dir / "final")


def _save_checkpoint(model: MultimodalAlignmentModel, path: Path, cfg: DictConfig) -> None:
    """Save the trainable VE state dict and projection heads.
    The frozen reference VE and CLIP text encoder are *not* saved (recoverable from HF).

    TODO(stage2): also write a tiny ``visual_only.safetensors`` ready to be dropped into
    a Qwen3-VL HF checkpoint dir's ``.visual.*`` keys for the veRL handoff.
    """
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trainable_visual": model.trainable_visual.state_dict(),
        "proj_img": model.proj_img.state_dict(),
        "proj_ts_img": model.proj_ts_img.state_dict(),
        "proj_ref": model.proj_ref.state_dict(),
        "proj_ref_ts": model.proj_ref_ts.state_dict(),
        "proj_text": model.proj_text.state_dict(),
        "log_logit_scale": model.log_logit_scale.detach().cpu(),
    }, path / "alignment_state.pt")
    OmegaConf.save(cfg, path / "config.yaml")
    logger.info("checkpoint saved to %s", path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 multimodal alignment trainer")
    p.add_argument("--config", "-c", required=True, type=str,
                   help="Path to YAML config (see verl/trainer/alignment/config/stage1_qwen3vl_clip.yaml)")
    p.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    train(cfg)


if __name__ == "__main__":
    main()
