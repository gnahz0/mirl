# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 CUDA runtime, data/model construction, and checkpoints."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data import AlignmentDataset, FamilyBalancedBatchSampler, collate_alignment
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


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def init_distributed() -> tuple[int, int, int, dist.ProcessGroup]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
        timeout=timedelta(minutes=60),
    )
    control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
    return rank, local_rank, world_size, control_group


def allreduce_grad_average(params: list[torch.nn.Parameter], world_size: int) -> None:
    if world_size == 1:
        return
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.wandb.enable:
        return None
    import wandb

    run = wandb.init(
        project=str(cfg.wandb.project),
        name=str(cfg.wandb.name),
        entity=str(cfg.wandb.get("entity")) if cfg.wandb.get("entity") else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(console="off"),
    )
    logger.info("W&B run initialized: %s", run.url)
    return run


def _dataset(
    cfg: DictConfig,
    files: list[str],
    seed: int,
    max_samples: int,
) -> AlignmentDataset:
    return AlignmentDataset(
        data_files=files,
        max_samples=max_samples,
        seed=seed,
        max_video_frames=cfg.data.max_video_frames,
    )


def build_loaders(
    cfg: DictConfig,
    rank: int,
    world_size: int,
    seed: int,
) -> tuple[
    AlignmentDataset,
    DataLoader,
    FamilyBalancedBatchSampler,
    DataLoader,
]:
    started = time.time()
    train_ds = _dataset(
        cfg,
        list(cfg.data.train_files),
        seed,
        cfg.data.get("max_train_samples", -1),
    )
    logger.info("train dataset: %d rows (%.1fs)", len(train_ds), time.time() - started)

    train_sampler = FamilyBalancedBatchSampler(
        train_ds,
        batch_size=cfg.train.batch_size,
        ts_per_family=cfg.data.ts_per_family_per_batch,
        rank=rank,
        world_size=world_size,
        seed=seed,
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
            multiprocessing_context=mp.get_context("spawn"),
            persistent_workers=True,
            prefetch_factor=cfg.train.prefetch_factor,
        )
    train_loader = DataLoader(train_ds, **train_kwargs)

    val_batches = cfg.train.val_batches
    started = time.time()
    val_ds = _dataset(
        cfg,
        list(cfg.data.val_files),
        seed,
        cfg.data.get("max_val_samples", -1),
    )
    val_sampler = FamilyBalancedBatchSampler(
        val_ds,
        batch_size=cfg.train.val_batch_size,
        ts_per_family=cfg.data.val_ts_per_family_per_batch,
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
        "val dataset: %d rows, batch/rank=%d, batches/eval=%s (%.1fs)",
        len(val_ds),
        cfg.train.val_batch_size,
        "all" if val_batches is None else val_batches,
        time.time() - started,
    )
    return train_ds, train_loader, train_sampler, val_loader


def build_model(
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
) -> MultimodalAlignmentModel:
    started = time.time()
    model = MultimodalAlignmentModel(
        qwen35_path=str(cfg.model.qwen35_path),
        siglip2_text_path=str(cfg.model.siglip2_text_path),
        shared_dim=cfg.projection.shared_dim,
        visual_dtype=visual_dtype,
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        ecg_normalization=cfg.model.ecg_normalization,
        tactile_delta_channels=cfg.model.tactile_delta_channels,
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
    head_lr = float(cfg.train.head_lr)
    scalar_lr = float(cfg.train.scalar_lr)
    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            head_lr=head_lr,
            scalar_lr=scalar_lr,
        ),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    warmup = math.ceil(total_steps * float(cfg.train.warmup_ratio))
    logger.info("cosine schedule: %d warmup steps, %d total optimizer steps", warmup, total_steps)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    return optimizer, scheduler


def save_checkpoint(model: MultimodalAlignmentModel, path: Path, cfg: DictConfig, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "trainable_visual": model.trainable_visual.state_dict(),
        "proj_visual": model.proj_visual.state_dict(),
        "proj_text": model.proj_text.state_dict(),
        "log_logit_scale": model.log_logit_scale.detach().cpu(),
        "step": step,
    }
    torch.save(state, path / "alignment_state.pt")
    OmegaConf.save(cfg, path / "config.yaml")
    logger.info("checkpoint saved to %s (step=%d)", path, step)


def load_checkpoint(model: MultimodalAlignmentModel, path: Path) -> int:
    ckpt_file = path / "alignment_state.pt" if path.is_dir() else path
    state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    model.trainable_visual.load_state_dict(state["trainable_visual"])
    model.proj_visual.load_state_dict(state["proj_visual"])
    model.proj_text.load_state_dict(state["proj_text"])
    with torch.no_grad():
        model.log_logit_scale.copy_(state["log_logit_scale"].to(model.log_logit_scale.device))
    logger.info("loaded %s at step %d", ckpt_file, state["step"])
    return state["step"]
