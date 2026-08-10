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
    _COUNT_KEYS,
    _REDUCED_METRIC_KEYS,
    _TS_FAMILIES,
    _metric_groups,
    all_reduce_sum,
    build_bank_specs,
    new_stats,
    prediction_metrics,
    update_stats,
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
    """One window's running loss sums, sample counts, and prediction statistics.

    Every published statistic is a ratio of sums over rows, so a microbatch folds
    into fixed float64 accumulators as it is seen and the window keeps no
    embeddings. The key set is fixed in ``__post_init__``, before any data, so the
    single all-reduce in ``flush()`` is identical on every rank.
    """

    device: torch.device
    label_bank: dict = field(default_factory=dict)
    tactile_bank: tuple | None = None
    log_logit_scale: torch.Tensor | None = None
    specs: tuple = field(init=False)
    stats: dict[str, torch.Tensor] = field(init=False)
    loss_sums: dict[str, float] = field(init=False)
    loss_counts: dict[str, int] = field(init=False)
    counts: dict[str, int] = field(init=False)
    local_values: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.specs = build_bank_specs(self.label_bank, self.tactile_bank)
        self.stats = new_stats(self.specs, self.device)
        self.loss_sums = dict.fromkeys(_REDUCED_METRIC_KEYS, 0.0)
        self.loss_counts = dict.fromkeys(_REDUCED_METRIC_KEYS, 0)
        self.counts = dict.fromkeys(_COUNT_KEYS, 0)

    def add(self, batch: dict, metrics: dict, ts_eval: tuple) -> None:
        for kind, value in batch.get("skipped", {}).items():
            self.counts[f"n/skipped_{kind}"] += int(value)
        size = len(batch["media"])
        if batch["kind"] == "signal":
            self.counts[f"n/ts_{batch['family']}"] += size
        else:
            self.counts[f"n/img_{batch['kind']}"] += size

        for key, value in metrics.items():
            if key in self.loss_sums:
                self.loss_sums[key] += float(value)
                self.loss_counts[key] += 1
            elif key.startswith("loss/"):
                raise RuntimeError(f"metric absent from _REDUCED_METRIC_KEYS: {key}")
            else:
                # grad_norm / logit_scale are rank-local diagnostics, not reduced.
                self.local_values.setdefault(key, []).append(float(value))

        update_stats(self.stats, self.specs, ts_eval, self.log_logit_scale)

    def flush(self, world_size: int = 1, per_class=None, per_label=None):
        """Reduce the whole window in one collective, then derive every number.

        Losses are averaged over the rank/microbatch pairs that produced them and
        dropped when nobody did.
        """
        payload = dict(self.stats)
        payload["_loss_sum"] = torch.tensor(
            [self.loss_sums[key] for key in _REDUCED_METRIC_KEYS],
            dtype=torch.float64,
            device=self.device,
        )
        payload["_loss_count"] = torch.tensor(
            [float(self.loss_counts[key]) for key in _REDUCED_METRIC_KEYS],
            dtype=torch.float64,
            device=self.device,
        )
        payload["_count"] = torch.tensor(
            [float(self.counts[key]) for key in _COUNT_KEYS],
            dtype=torch.float64,
            device=self.device,
        )
        reduced = all_reduce_sum(payload, world_size=world_size)

        sums = dict(zip(_REDUCED_METRIC_KEYS, reduced["_loss_sum"].tolist(), strict=True))
        weights = dict(zip(_REDUCED_METRIC_KEYS, reduced["_loss_count"].tolist(), strict=True))
        losses = {key: sums[key] / weights[key] for key in _REDUCED_METRIC_KEYS if weights[key]}
        losses.update({key: sum(v) / len(v) for key, v in self.local_values.items()})
        counts = {
            key: int(value) for key, value in zip(_COUNT_KEYS, reduced["_count"].tolist(), strict=True)
        }
        counts["n/ts_signal"] = sum(counts[f"n/ts_{family}"] for family in _TS_FAMILIES)
        prediction = prediction_metrics(reduced, self.specs, per_class=per_class, per_label=per_label)
        return losses, counts, prediction


@torch.no_grad()
def _run_validation(model, val_loader, cfg, accelerator, label_bank, tactile_bank):
    """Evaluate the full sharded validation set."""
    was_training = model.training
    model.eval()
    world_size = accelerator.num_processes
    base_model = accelerator.unwrap_model(model)
    window = AccumulationWindow(
        accelerator.device, label_bank, tactile_bank, base_model.log_logit_scale
    )
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

    per_class: dict[str, list[dict[str, object]]] = {}
    per_label: list[dict[str, object]] = []
    losses, counts, prediction = window.flush(world_size, per_class, per_label)

    if was_training:
        model.train()
        base_model.frozen_visual.eval()
        base_model.label_text_model.eval()
    return _metric_groups("val", losses, counts, prediction), per_class, per_label


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
    window_args = (device, train_label_bank, tactile_bank, base_model.log_logit_scale)
    window = AccumulationWindow(*window_args)
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

            window.local_values.setdefault("grad_norm", []).append(
                float(accelerator.clip_grad_norm_(trainable, cfg.train.grad_clip))
            )
            metrics, counts, window_metrics = window.flush(world_size)
            optimizer.step()
            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

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
                        "train-aux/lr": lr,
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

            window = AccumulationWindow(*window_args)

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
