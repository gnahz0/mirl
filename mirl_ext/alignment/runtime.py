# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 CUDA runtime, data/model construction, and checkpoints."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .data import AlignmentDataset, HomogeneousBatchSampler, collate_alignment
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in ("qwen_vl_utils", "qwen_vl_utils.vision_process", "torchcodec"):
        logging.getLogger(name).setLevel(logging.WARNING)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def build_loaders(
    cfg: DictConfig,
    rank: int,
    world_size: int,
    seed: int,
) -> tuple[
    AlignmentDataset,
    AlignmentDataset,
    DataLoader,
    HomogeneousBatchSampler,
    DataLoader,
]:
    train_ds = AlignmentDataset(
        list(cfg.data.train_files),
        max_video_frames=cfg.data.max_video_frames,
    )

    train_sampler = HomogeneousBatchSampler(
        train_ds,
        batch_size=cfg.train.batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        signal_repeat_factors=dict(cfg.train.signal_repeat_factors),
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_alignment,
        pin_memory=True,
        multiprocessing_context="spawn" if cfg.train.num_workers else None,
        persistent_workers=bool(cfg.train.num_workers),
    )

    val_ds = AlignmentDataset(
        list(cfg.data.val_files),
        max_video_frames=cfg.data.max_video_frames,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=HomogeneousBatchSampler(
            val_ds,
            batch_size=cfg.train.val_batch_size,
            rank=rank,
            world_size=world_size,
            seed=seed + 1,
        ),
        num_workers=0,
        collate_fn=collate_alignment,
        pin_memory=True,
    )

    return train_ds, val_ds, train_loader, train_sampler, val_loader


def build_optimizer(model: MultimodalAlignmentModel, cfg: DictConfig, total_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    # One learning rate for everything, temperature included, as SigLIP does.
    # 0-dim scalars fail an `ndim == 1` predicate, so the no-decay split is `<= 1`.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in trainable if p.ndim > 1],
                "weight_decay": cfg.train.weight_decay,
            },
            {
                "params": [p for p in trainable if p.ndim <= 1],
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer, get_cosine_schedule_with_warmup(
        optimizer,
        math.ceil(total_steps * float(cfg.train.warmup_ratio)),
        total_steps,
    )


def load_checkpoint(model: MultimodalAlignmentModel, path: str | Path) -> None:
    """Warm-start trainable model weights from an alignment checkpoint."""
    state_path = Path(path) / "alignment_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.trainable_visual.load_state_dict(state["trainable_visual"], strict=True)
    with torch.no_grad():
        model.log_logit_scale.copy_(state["log_logit_scale"])
        model.logit_bias.copy_(state["logit_bias"])
    step = int(state["step"])
    logger.info("loaded alignment checkpoint %s (step=%d)", state_path, step)


def load_training_state(path: str | Path, optimizer, scheduler) -> dict[str, Any]:
    """Restore optimizer/scheduler state saved at an optimizer-step boundary."""
    state_path = Path(path) / "trainer_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    logger.info("loaded trainer state %s (step=%d)", state_path, int(state["step"]))
    return state


def save_checkpoint(
    model: MultimodalAlignmentModel,
    path: Path,
    cfg: DictConfig,
    step: int,
    *,
    optimizer=None,
    scheduler=None,
    progress: dict[str, Any] | None = None,
) -> None:
    """Save model weights and, when supplied, resumable trainer state."""
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "trainable_visual": model.trainable_visual.state_dict(),
        "log_logit_scale": model.log_logit_scale.detach().cpu(),
        "logit_bias": model.logit_bias.detach().cpu(),
        "step": step,
    }
    torch.save(state, path / "alignment_state.pt")
    if optimizer is not None:
        trainer_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            **(progress or {}),
        }
        torch.save(trainer_state, path / "trainer_state.pt")
    OmegaConf.save(cfg, path / "config.yaml")
    logger.info(
        "checkpoint saved to %s (step=%d, trainer_state=%s)",
        path,
        step,
        optimizer is not None,
    )
