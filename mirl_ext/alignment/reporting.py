# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Console and W&B reporting for Stage-1 alignment."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from .metrics import _METRIC_FAMILY_NAMES, _training_metric_groups
from .objective import _run_validation
from .runtime import save_checkpoint


def report_train_step(
    model,
    cfg: DictConfig,
    out_dir: Path,
    wandb_run,
    pbar: tqdm,
    metrics: dict[str, float],
    counts: dict[str, int],
    lrs: dict[str, float],
    step: int,
    started: float,
    validate_now: bool,
    epoch: float,
) -> None:
    public = _training_metric_groups(metrics, counts)
    pbar.set_postfix(
        loss=f"{metrics['loss/total']:.3f}",
        lr=f"{lrs['vit']:.1e}",
        img=counts["n/img_image"] + counts["n/img_video"],
        ts=counts["n/ts_signal"],
        refresh=False,
    )
    pbar.update()

    if wandb_run is not None:
        wandb_run.log(
            {
                **public,
                "train-aux/lr/vit": lrs["vit"],
                "train-aux/lr/head": lrs["head"],
                "train-aux/lr/scalar": lrs["scalar"],
                "train-aux/epoch": epoch,
            },
            step=step,
            commit=not validate_now,
        )

    if step % cfg.train.log_every == 0:
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        core = " ".join(f"{key}={value:.3f}" for key, value in public.items() if key.startswith("train-core/"))
        tqdm.write(
            f"step {step:6d} | lr {lrs['vit']:.2e} | loss={metrics['loss/total']:.4f} "
            f"| {core} | peak {peak_gib:.1f}GiB | {time.time() - started:.1f}s",
            file=sys.stdout,
        )

    if step % cfg.train.ckpt_every == 0:
        save_checkpoint(model, out_dir / "latest", cfg, step)


def report_validation(
    model,
    val_loader,
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
    amp_dtype: torch.dtype,
    prototype_bank,
    gcms_bank,
    out_dir: Path,
    wandb_run,
    step: int,
    best_value: float,
) -> float:
    metrics, per_class = _run_validation(
        model,
        val_loader,
        cfg,
        device,
        visual_dtype,
        amp_dtype,
        n_batches=cfg.train.val_batches,
        prototype_bank=prototype_bank,
        gcms_bank=gcms_bank,
    )
    core = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()) if key.startswith("val-core/"))
    aux = " ".join(
        f"{key}={value:.4f}"
        for key, value in sorted(metrics.items())
        if key == "val/loss"
        or key.startswith(("val-aux/effective_dimension/", "val-aux/prediction_coverage/"))
    )
    tqdm.write(f"VAL-CORE @ step {step:6d} | {core}", file=sys.stdout)
    tqdm.write(f"VAL-AUX  @ step {step:6d} | {aux}", file=sys.stdout)

    current = metrics[cfg.train.best_metric]
    is_best = current > best_value
    if is_best:
        best_value = current
        save_checkpoint(model, out_dir / "best", cfg, step)

    if wandb_run is not None:
        import wandb

        payload = dict(metrics)
        if is_best:
            payload.update({"best/metric": current, "best/step": step})
        columns = ("class_id", "label", "support", "predicted", "precision", "recall", "f1")
        for family, rows in per_class.items():
            payload[f"val-aux/per_class/{_METRIC_FAMILY_NAMES[family]}"] = wandb.Table(
                columns=list(columns),
                data=[[row[column] for column in columns] for row in rows],
            )
        wandb_run.log(payload, step=step, commit=True)
    return best_value
