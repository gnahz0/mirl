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
    _merge_prediction_metrics,
    _metric_groups,
    _tactile_prediction_metrics,
    _ts_prediction_metrics,
    add_batch_counts,
)
from .objective import (
    _build_tactile_bank,
    _build_text_label_bank,
    _compute_losses,
)
from .runtime import (
    build_loaders,
    build_model,
    build_optimizer,
    load_checkpoint,
    load_training_state,
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
    tactile_embeddings: list[torch.Tensor] = field(default_factory=list)
    tactile_target_chunks: list[torch.Tensor] = field(default_factory=list)
    tactile_mask_chunks: list[torch.Tensor] = field(default_factory=list)

    def add(self, batch: dict, metrics: dict, ts_eval: tuple) -> None:
        add_batch_counts(self.counts, batch)
        embeddings, labels, families, tactile_targets, tactile_masks = ts_eval
        if embeddings is not None:
            if tactile_targets is not None:
                self.tactile_embeddings.append(embeddings)
                self.tactile_target_chunks.append(tactile_targets)
                self.tactile_mask_chunks.append(tactile_masks)
            else:
                self.ts_embeddings.append(embeddings)
                self.ts_labels.extend(labels)
                self.ts_families.extend(families)
        for key, value in metrics.items():
            self.metric_values.setdefault(key, []).append(float(value))

    def score(
        self,
        label_bank,
        tactile_bank,
        logit_scale,
        world_size: int,
        per_class=None,
        per_label=None,
    ) -> dict[str, float]:
        standard = (
            _ts_prediction_metrics(
                torch.cat(self.ts_embeddings),
                self.ts_labels,
                self.ts_families,
                label_bank,
                world_size=world_size,
                per_class_reports=per_class,
            )
            if self.ts_embeddings
            else {}
        )
        tactile = (
            _tactile_prediction_metrics(
                torch.cat(self.tactile_embeddings),
                torch.cat(self.tactile_target_chunks),
                torch.cat(self.tactile_mask_chunks),
                tactile_bank,
                logit_scale,
                world_size=world_size,
                per_label_out=per_label,
            )
            if self.tactile_embeddings
            else {}
        )
        return _merge_prediction_metrics(standard, tactile)


@torch.no_grad()
def _run_validation(model, val_loader, cfg, accelerator, label_bank, tactile_bank):
    """Evaluate the full sharded validation set."""
    was_training = model.training
    model.eval()
    world_size = accelerator.num_processes
    window = AccumulationWindow()
    for batch in val_loader:
        with accelerator.autocast():
            _, metrics, batch_eval = _compute_losses(
                model,
                batch,
                cfg,
                world_size=world_size,
                label_bank=label_bank,
                tactile_bank=tactile_bank,
            )
        window.add(batch, metrics, batch_eval)

    losses = _allreduce_metrics(
        {key: sum(values) / len(values) for key, values in window.metric_values.items()},
        accelerator.device,
        world_size,
    )
    counts = _allreduce_counts(window.counts, accelerator.device, world_size)
    per_class: dict[str, list[dict[str, object]]] = {}
    per_label: list[dict[str, object]] = []
    base_model = accelerator.unwrap_model(model)
    prediction_metrics = window.score(
        label_bank,
        tactile_bank,
        base_model.log_logit_scale,
        world_size,
        per_class,
        per_label,
    )

    if was_training:
        model.train()
        base_model.frozen_visual.eval()
        base_model.label_text_model.eval()
    return _metric_groups("val", losses, counts, prediction_metrics), per_class, per_label


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

    init_checkpoint = cfg.train.get("init_checkpoint")
    resume_checkpoint = cfg.train.get("resume_checkpoint")
    if init_checkpoint and resume_checkpoint:
        raise ValueError("train.init_checkpoint and train.resume_checkpoint are mutually exclusive")

    train_ds, val_ds, train_loader, train_sampler, val_loader = build_loaders(cfg, rank, world_size, seed)
    model = build_model(cfg, device, visual_dtype)
    if init_checkpoint or resume_checkpoint:
        load_checkpoint(model, str(init_checkpoint or resume_checkpoint))
        if init_checkpoint:
            logger.info("warm start uses a fresh optimizer and schedule")
    # Encode each split's complete label banks once.
    train_label_bank = _build_text_label_bank(model, train_ds.ts_label_vocabs, device)
    val_label_bank = _build_text_label_bank(model, val_ds.ts_label_vocabs, device)
    tactile_bank = _build_tactile_bank(
        model,
        train_ds.task_positive_rates,
        device,
    )
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
    resume_progress = None
    if resume_checkpoint:
        resume_progress = load_training_state(str(resume_checkpoint), optimizer, scheduler)
        saved_steps_per_epoch = int(resume_progress["steps_per_epoch"])
        saved_total_steps = int(resume_progress["total_steps"])
        if saved_steps_per_epoch != steps_per_epoch or saved_total_steps != total_steps:
            raise ValueError(
                "resume checkpoint schedule does not match this run: "
                f"steps_per_epoch {saved_steps_per_epoch} != {steps_per_epoch} or "
                f"total_steps {saved_total_steps} != {total_steps}; use "
                "train.init_checkpoint for a weights-only continuation"
            )
    if is_main:
        import wandb

        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            name=str(cfg.wandb.name),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(console="off"),
        )
        logger.info("W&B run initialized: %s", wandb_run.url)
    else:
        wandb_run = None
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
    opt_step = int(resume_progress["step"]) if resume_progress else 0
    best_value = float(resume_progress.get("best_value", float("-inf"))) if resume_progress else float("-inf")
    start_epoch = int(resume_progress.get("next_epoch", 0)) if resume_progress else 0
    start_batch_index = int(resume_progress.get("next_batch_index", 0)) if resume_progress else 0
    if start_batch_index % grad_accum:
        raise ValueError("resume checkpoint is not at an optimizer-step boundary")
    if start_epoch >= num_epochs:
        raise ValueError(
            f"resume checkpoint already completed epoch {start_epoch} of {num_epochs}; "
            "use train.init_checkpoint to start a new schedule"
        )
    if resume_progress:
        logger.info(
            "resuming at optimizer step %d, epoch %d, microbatch %d",
            opt_step,
            start_epoch,
            start_batch_index,
        )
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
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
                        tactile_bank=tactile_bank,
                    )
                window.add(batch, metrics, micro_eval)
                accelerator.backward(loss / window_size)

            if not sync_now:
                continue

            metrics = {key: sum(values) / len(values) for key, values in window.metric_values.items()}
            metrics["grad_norm"] = float(accelerator.clip_grad_norm_(trainable, cfg.train.grad_clip))
            window_metrics = window.score(
                train_label_bank,
                tactile_bank,
                base_model.log_logit_scale,
                world_size,
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
                payload = _metric_groups("train", metrics, counts, window_metrics)
                skipped_total = sum(cumulative_counts[f"n/skipped_{kind}"] for kind in ("image", "video", "signal"))
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
                val_metrics, per_class, per_label = _run_validation(
                    model,
                    val_loader,
                    cfg,
                    accelerator,
                    label_bank=val_label_bank,
                    tactile_bank=tactile_bank,
                )
                if is_main:
                    current = val_metrics[cfg.train.best_metric]
                    is_best = current > best_value
                    if is_best:
                        best_value = current
                        save_checkpoint(base_model, out_dir / "best", cfg, opt_step)
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
                    tactile_columns = (
                        "task",
                        "label",
                        "support",
                        "predicted",
                        "precision",
                        "recall",
                        "f1",
                    )
                    if per_label:
                        payload["val-aux/per_label/tactile"] = wandb.Table(
                            columns=list(tactile_columns),
                            data=[[row[column] for column in tactile_columns] for row in per_label],
                        )
                    wandb_run.log(payload, step=opt_step, commit=True)
                    save_checkpoint(
                        base_model,
                        out_dir / "last",
                        cfg,
                        opt_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress={
                            "best_value": best_value,
                            "next_epoch": epoch + int(end_of_epoch),
                            "next_batch_index": 0 if end_of_epoch else batch_index + 1,
                            "steps_per_epoch": steps_per_epoch,
                            "total_steps": total_steps,
                        },
                    )
                accelerator.wait_for_everyone()

            window = AccumulationWindow()

    if is_main:
        save_checkpoint(base_model, out_dir / "final", cfg, opt_step)
        wandb_run.finish()
        logger.info(
            "training complete: best %s=%.4f; skipped=%s; final=%s",
            cfg.train.best_metric,
            best_value,
            {kind: cumulative_counts[f"n/skipped_{kind}"] for kind in ("image", "video", "signal")},
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
