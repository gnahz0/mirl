# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Train Qwen3.5's vision encoder against SigLIP2 text prototypes."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .metrics import (
    _allreduce_counts,
    _allreduce_metrics,
    _is_ts_window_metric,
    add_ts_family_counts,
    group_grad_norms,
    new_counts,
)
from .objective import _build_text_prototype_bank, _compute_losses, _score_ts_collector
from .reporting import report_train_step, report_validation
from .runtime import (
    allreduce_grad_average,
    build_loaders,
    build_model,
    build_optimizer,
    init_distributed,
    log_param_summary,
    maybe_init_wandb,
    resolve_dtype,
    save_checkpoint,
    setup_logging,
)

logger = logging.getLogger("alignment.trainer")


def train(cfg: DictConfig) -> None:
    setup_logging(cfg.get("log_level", "INFO"))
    # torchrun owns one full model replica per GPU; gradients are synced manually.
    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0
    seed = int(cfg.train.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    device = torch.device("cuda", local_rank)
    amp_dtype = resolve_dtype(cfg.train.amp_dtype)
    visual_dtype = resolve_dtype(cfg.model.trainable_visual_dtype)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Stage 1 on rank %d/%d, %s (%.1f GiB), output=%s",
        rank,
        world_size,
        torch.cuda.get_device_name(local_rank),
        torch.cuda.get_device_properties(local_rank).total_memory / (1024**3),
        out_dir,
    )

    train_ds, train_loader, train_sampler, val_loader = build_loaders(cfg, rank, world_size, seed)
    model, gcms_bank = build_model(cfg, train_ds, device, visual_dtype)
    # The complete SigLIP2 label vocabulary acts as the classifier; no class head.
    families = tuple(cfg.loss.prototype_families)
    prototype_bank = _build_text_prototype_bank(
        model,
        {family: train_ds.ts_label_vocabs[family] for family in families},
        device,
    )
    if is_main:
        log_param_summary(model)
    free, total = torch.cuda.mem_get_info()
    logger.info(
        "GPU memory after load: %.2f / %.2f GiB",
        (total - free) / (1024**3),
        total / (1024**3),
    )

    grad_accum = int(cfg.train.get("grad_accum_steps", 1))
    num_epochs = int(cfg.train.num_train_epochs)
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum)
    total_steps = num_epochs * steps_per_epoch
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)
    wandb_run = maybe_init_wandb(cfg) if is_main else None
    model.train()
    model.frozen_visual.eval()
    model.label_text_model.eval()
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    effective_batch = cfg.train.batch_size * world_size * grad_accum
    logger.info(
        "training %d epoch(s), %d microbatches/epoch, %d optimizer steps/epoch "
        "(%d total); batch/rank=%d, GPUs=%d, accumulation=%d, effective batch=%d",
        num_epochs,
        micro_batches_per_epoch,
        steps_per_epoch,
        total_steps,
        cfg.train.batch_size,
        world_size,
        grad_accum,
        effective_batch,
    )

    pbar = tqdm(total=total_steps, desc="stage1", dynamic_ncols=True, file=sys.stdout) if is_main else None
    trainable = [param for param in model.parameters() if param.requires_grad]
    counts = new_counts()
    metric_sums: dict[str, float] = {}
    metric_observations: dict[str, int] = {}
    # Keep embeddings from the exact microbatches contributing to one update.
    ts_eval = {"z": [], "labels": [], "families": []} if is_main else None
    opt_step = 0
    best_value = float("-inf")
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            window_start = (batch_index // grad_accum) * grad_accum
            window_size = min(grad_accum, micro_batches_per_epoch - window_start)
            micro_eval = {"z": [], "labels": [], "families": []} if is_main else None
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, metrics = _compute_losses(
                    model,
                    batch,
                    cfg,
                    device,
                    visual_dtype,
                    world_size=world_size,
                    ts_eval_collector=micro_eval,
                    prototype_bank=prototype_bank,
                    gcms_bank=gcms_bank,
                )

            counts["n/img_image"] += len(batch["img_image_pil"])
            counts["n/img_video"] += len(batch["img_video"])
            counts["n/ts_signal"] += len(batch["ts_signal"])
            add_ts_family_counts(counts, batch)
            if is_main:
                ts_eval["z"].extend(micro_eval["z"])
                ts_eval["labels"].extend(micro_eval["labels"])
                ts_eval["families"].extend(micro_eval["families"])
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_observations[key] = metric_observations.get(key, 0) + 1

            (loss / window_size).backward()
            end_of_epoch = batch_index + 1 == micro_batches_per_epoch
            if (batch_index + 1) % grad_accum and not end_of_epoch:
                continue

            metrics = {key: total / metric_observations[key] for key, total in metric_sums.items()}
            # Score before optimizer.step so embeddings and prototypes use one model state.
            window_metrics = (
                _score_ts_collector(
                    model,
                    ts_eval,
                    prototype_bank,
                    gcms_bank=gcms_bank,
                )
                if is_main
                else {}
            )
            # Average first, then clip, so every replica applies the same update.
            allreduce_grad_average(trainable, world_size)
            metrics.update(group_grad_norms(model))
            metrics["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip))
            optimizer.step()
            lrs = {str(group["name"]).partition("_")[0]: float(group["lr"]) for group in optimizer.param_groups}
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            metrics = _allreduce_metrics(metrics, device, world_size)
            counts = _allreduce_counts(counts, device, world_size)
            validate_now = opt_step % cfg.train.val_every == 0 or end_of_epoch
            if is_main:
                metrics = {key: value for key, value in metrics.items() if not _is_ts_window_metric(key)}
                metrics.update(window_metrics)
                report_train_step(
                    model,
                    cfg,
                    out_dir,
                    wandb_run,
                    pbar,
                    metrics,
                    counts,
                    lrs,
                    opt_step,
                    started,
                    validate_now,
                    epoch + (batch_index + 1) / micro_batches_per_epoch,
                )

            if validate_now:
                # Other ranks wait while rank 0 evaluates the fixed validation subset.
                dist.barrier()
                if is_main:
                    best_value = report_validation(
                        model,
                        val_loader,
                        cfg,
                        device,
                        visual_dtype,
                        amp_dtype,
                        prototype_bank,
                        gcms_bank,
                        out_dir,
                        wandb_run,
                        opt_step,
                        best_value,
                    )
                dist.barrier()

            counts = new_counts()
            metric_sums = {}
            metric_observations = {}
            ts_eval = {"z": [], "labels": [], "families": []} if is_main else None

    if is_main:
        pbar.close()
        save_checkpoint(model, out_dir / "final", cfg, opt_step)
        if wandb_run is not None:
            wandb_run.finish()
        logger.info(
            "training complete: best %s=%.4f; final=%s",
            cfg.train.best_metric,
            best_value,
            out_dir / "final",
        )
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(args.overrides),
    )
    train(cfg)


if __name__ == "__main__":
    main()
