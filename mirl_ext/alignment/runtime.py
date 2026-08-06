# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 CUDA runtime, data/model construction, and checkpoints."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

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


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb.enable:
        return None
    import wandb

    run = wandb.init(
        project=str(cfg.wandb.project),
        name=str(cfg.wandb.name),
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(console="off"),
    )
    logger.info("W&B run initialized: %s", run.url)
    return run


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
    started = time.time()
    train_ds = AlignmentDataset(
        list(cfg.data.train_files),
        max_video_frames=cfg.data.max_video_frames,
    )
    logger.info("train dataset: %d rows (%.1fs)", len(train_ds), time.time() - started)

    train_sampler = HomogeneousBatchSampler(
        train_ds,
        batch_size=cfg.train.batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        signal_repeat_factors=dict(cfg.train.get("signal_repeat_factors", {})),
    )
    train_kwargs = {
        "batch_sampler": train_sampler,
        "num_workers": cfg.train.num_workers,
        "collate_fn": collate_alignment,
        "pin_memory": True,
    }
    if cfg.train.num_workers:
        # CUDA is already initialized, so workers must spawn rather than fork.
        train_kwargs.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    train_loader = DataLoader(train_ds, **train_kwargs)

    started = time.time()
    val_ds = AlignmentDataset(
        list(cfg.data.val_files),
        max_video_frames=cfg.data.max_video_frames,
    )
    val_sampler = HomogeneousBatchSampler(
        val_ds,
        batch_size=cfg.train.val_batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed + 1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=0,
        collate_fn=collate_alignment,
        pin_memory=True,
    )
    logger.info(
        "val dataset: %d rows, batch/rank=%d, full evaluation (%.1fs)",
        len(val_ds),
        cfg.train.val_batch_size,
        time.time() - started,
    )
    return train_ds, val_ds, train_loader, train_sampler, val_loader


def build_model(
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
) -> MultimodalAlignmentModel:
    started = time.time()
    model = MultimodalAlignmentModel(
        qwen35_path=str(cfg.model.qwen35_path),
        siglip2_text_path=str(cfg.model.siglip2_text_path),
        visual_dtype=visual_dtype,
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        contrastive_temperature=cfg.loss.temperature,
    ).to(device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    # Keep fp32 master weights while autocast handles bf16 forward operations.
    for param in trainable:
        param.data = param.data.float()
    logger.info(
        "model ready in %.1fs; %.1fM trainable parameters",
        time.time() - started,
        sum(param.numel() for param in trainable) / 1e6,
    )
    return model


def build_optimizer(model: MultimodalAlignmentModel, cfg: DictConfig, total_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad and p is not model.log_logit_scale]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "model_decay",
                "params": [p for p in trainable if p.ndim > 1],
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
            },
            {
                "name": "model_no_decay",
                "params": [p for p in trainable if p.ndim <= 1],
                "lr": cfg.train.lr,
                "weight_decay": 0.0,
            },
            {
                "name": "scalar",
                "params": [model.log_logit_scale],
                "lr": cfg.train.scalar_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    warmup = math.ceil(total_steps * float(cfg.train.warmup_ratio))
    logger.info("cosine schedule: %d warmup steps, %d total optimizer steps", warmup, total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    return optimizer, scheduler


def save_checkpoint(model: MultimodalAlignmentModel, path: Path, cfg: DictConfig, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "trainable_visual": model.trainable_visual.state_dict(),
        "log_logit_scale": model.log_logit_scale.detach().cpu(),
        "step": step,
    }
    torch.save(state, path / "alignment_state.pt")
    OmegaConf.save(cfg, path / "config.yaml")
    logger.info("checkpoint saved to %s (step=%d)", path, step)
