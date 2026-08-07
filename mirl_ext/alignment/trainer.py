# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Train Qwen3.5's vision encoder against SigLIP2 text-label banks."""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from .metrics import (
    _allreduce_counts,
    _allreduce_metrics,
    _metric_groups,
    add_batch_counts,
)
from .objective import (
    _build_text_label_bank,
    _compute_losses,
    _run_validation,
    _score_ts,
)
from .runtime import (
    build_loaders,
    build_model,
    build_optimizer,
    maybe_init_wandb,
    save_checkpoint,
    setup_logging,
)

logger = logging.getLogger("alignment.trainer")


@dataclass
class AccumulationWindow:
    counts: Counter[str] = field(default_factory=Counter)
    metric_values: dict[str, list[float]] = field(default_factory=dict)
    ts_embeddings: list[torch.Tensor] = field(default_factory=list)
    ts_labels: list[str] = field(default_factory=list)
    ts_families: list[str] = field(default_factory=list)

    def add(self, batch: dict, metrics: dict, ts_eval: tuple) -> None:
        add_batch_counts(self.counts, batch)
        embeddings, labels, families = ts_eval
        if embeddings is not None:
            self.ts_embeddings.append(embeddings)
            self.ts_labels.extend(labels)
            self.ts_families.extend(families)
        for key, value in metrics.items():
            self.metric_values.setdefault(key, []).append(float(value))


def train(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    accelerator = Accelerator(mixed_precision=str(cfg.train.amp_dtype))
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process
    seed = int(cfg.train.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    device = accelerator.device
    # Use autocast for frozen towers and fp32 trainable weights.
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    visual_dtype = dtypes[str(cfg.train.amp_dtype)]
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Stage 1 on rank %d/%d, %s (%.1f GiB), output=%s",
        rank,
        world_size,
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_properties(device).total_memory / (1024**3),
        out_dir,
    )

    train_ds, val_ds, train_loader, train_sampler, val_loader = build_loaders(cfg, rank, world_size, seed)
    model = build_model(cfg, device, visual_dtype)
    # Encode each split's complete label banks once.
    train_label_bank = _build_text_label_bank(model, train_ds.ts_label_vocabs, device)
    val_label_bank = _build_text_label_bank(model, val_ds.ts_label_vocabs, device)
    free, total = torch.cuda.mem_get_info()
    logger.info(
        "GPU memory after load: %.2f / %.2f GiB",
        (total - free) / (1024**3),
        total / (1024**3),
    )

    grad_accum = int(cfg.train.grad_accum_steps)
    num_epochs = int(cfg.train.num_train_epochs)
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum)
    total_steps = num_epochs * steps_per_epoch
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)
    model, optimizer = accelerator.prepare(model, optimizer)
    base_model = accelerator.unwrap_model(model)
    wandb_run = maybe_init_wandb(cfg) if is_main else None
    model.train()
    base_model.frozen_visual.eval()
    base_model.label_text_model.eval()
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

    trainable = [param for param in model.parameters() if param.requires_grad]
    window = AccumulationWindow()
    cumulative_counts: Counter[str] = Counter()
    opt_step = 0
    best_value = float("-inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            window_start = (batch_index // grad_accum) * grad_accum
            window_size = min(grad_accum, micro_batches_per_epoch - window_start)
            end_of_epoch = batch_index + 1 == micro_batches_per_epoch
            sync_now = (batch_index + 1) % grad_accum == 0 or end_of_epoch
            sync_context = nullcontext() if sync_now else accelerator.no_sync(model)
            with sync_context:
                with accelerator.autocast():
                    loss, metrics, micro_eval = _compute_losses(
                        model,
                        batch,
                        cfg,
                        world_size=world_size,
                        label_bank=train_label_bank,
                    )
                window.add(batch, metrics, micro_eval)
                accelerator.backward(loss / window_size)

            if not sync_now:
                continue

            metrics = {key: sum(values) / len(values) for key, values in window.metric_values.items()}
            metrics["grad_norm"] = float(accelerator.clip_grad_norm_(trainable, cfg.train.grad_clip))
            window_metrics = _score_ts(
                window.ts_embeddings,
                window.ts_labels,
                window.ts_families,
                train_label_bank,
                world_size=world_size,
            )
            optimizer.step()
            lrs = {str(group["name"]).partition("_")[0]: float(group["lr"]) for group in optimizer.param_groups}
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            metrics = _allreduce_metrics(metrics, device, world_size)
            counts = _allreduce_counts(window.counts, device, world_size)
            cumulative_counts.update(counts)
            validate_now = opt_step % cfg.train.val_every == 0 or end_of_epoch
            if is_main:
                metrics.update(window_metrics)
                if wandb_run is not None:
                    payload = _metric_groups("train", metrics, counts)
                    skipped_total = sum(
                        cumulative_counts[f"n/skipped_{kind}"]
                        for kind in ("image", "video", "signal")
                    )
                    valid_total = (
                        cumulative_counts["n/img_image"]
                        + cumulative_counts["n/img_video"]
                        + cumulative_counts["n/ts_signal"]
                    )
                    for kind in ("image", "video", "signal"):
                        payload[f"train-aux/n/skipped_cumulative/{kind}"] = float(
                            cumulative_counts[f"n/skipped_{kind}"]
                        )
                    payload["train-aux/n/skipped_cumulative/total"] = float(skipped_total)
                    payload["train-aux/skipped_fraction_cumulative"] = skipped_total / max(
                        valid_total + skipped_total,
                        1,
                    )
                    wandb_run.log(
                        {
                            **payload,
                            "train-aux/lr/model": lrs["model"],
                            "train-aux/lr/scalar": lrs["scalar"],
                            "train-aux/epoch": epoch + (batch_index + 1) / micro_batches_per_epoch,
                        },
                        step=opt_step,
                        commit=not validate_now,
                    )

            if validate_now:
                val_metrics, per_class = _run_validation(
                    model,
                    val_loader,
                    cfg,
                    accelerator,
                    label_bank=val_label_bank,
                )
                if is_main:
                    current = val_metrics[cfg.train.best_metric]
                    is_best = current > best_value
                    if is_best:
                        best_value = current
                        save_checkpoint(base_model, out_dir / "best", cfg, opt_step)
                    if wandb_run is not None:
                        import wandb

                        payload = dict(val_metrics)
                        if is_best:
                            payload.update({"best/metric": current, "best/step": opt_step})
                        columns = (
                            "class_id",
                            "label",
                            "support",
                            "predicted",
                            "precision",
                            "recall",
                            "f1",
                            "recall_at_5",
                        )
                        for family, rows in per_class.items():
                            payload[f"val-aux/per_class/{family}"] = wandb.Table(
                                columns=list(columns),
                                data=[[row[column] for column in columns] for row in rows],
                            )
                        wandb_run.log(payload, step=opt_step, commit=True)
                accelerator.wait_for_everyone()

            window = AccumulationWindow()

    if is_main:
        save_checkpoint(base_model, out_dir / "final", cfg, opt_step)
        if wandb_run is not None:
            wandb_run.finish()
        logger.info(
            "training complete: best %s=%.4f; skipped=%s; final=%s",
            cfg.train.best_metric,
            best_value,
            {
                kind: cumulative_counts[f"n/skipped_{kind}"]
                for kind in ("image", "video", "signal")
            },
            out_dir / "final",
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


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
