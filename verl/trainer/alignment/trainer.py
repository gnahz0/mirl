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

import torch
import torch.nn.functional as F
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


def _qwen_process_batch(processor, pil_list, device, dtype):
    """Run the Qwen3-VL processor on a list of PIL images. The processor expects a
    matching ``text=`` argument; we use a single ``<image>``-padded prompt per image so
    grid_thw is computed correctly."""
    if not pil_list:
        return None
    text = ["<image>"] * len(pil_list)
    out = processor(images=pil_list, text=text, return_tensors="pt", padding=True)
    return {
        "pixel_values": out["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw": out["image_grid_thw"].to(device=device),
    }


def _compute_losses(
    model: MultimodalAlignmentModel,
    img_inputs,
    img_text,
    ts_inputs,
    ts_text,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    metrics: dict[str, float] = {}

    z_img = z_ts = z_ref_img = z_ref_ts = z_text_img = z_text_ts = None

    if img_inputs is not None:
        feat_img = model.encode_images_trainable(img_inputs["pixel_values"], img_inputs["image_grid_thw"])
        feat_ref_img = model.encode_images_frozen(img_inputs["pixel_values"], img_inputs["image_grid_thw"])
        z_img = model.project(model.proj_img, feat_img)
        z_ref_img = model.project(model.proj_ref, feat_ref_img)
        clip_feat_img = model.encode_text(img_text, device=device)
        z_text_img = model.project(model.proj_text, clip_feat_img.float())

    if ts_inputs is not None:
        feat_ts = model.encode_images_trainable(ts_inputs["pixel_values"], ts_inputs["image_grid_thw"])
        feat_ref_ts = model.encode_images_frozen(ts_inputs["pixel_values"], ts_inputs["image_grid_thw"])
        z_ts = model.project(model.proj_ts_img, feat_ts)
        z_ref_ts = model.project(model.proj_ref_ts, feat_ref_ts)
        clip_feat_ts = model.encode_text(ts_text, device=device)
        z_text_ts = model.project(model.proj_text, clip_feat_ts.float())

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
    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=str(cfg.wandb.get("project", "mirl-alignment")),
                name=str(cfg.wandb.get("name", "stage1")),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("wandb init failed (%s); continuing without it", e)

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
            img_inputs = _qwen_process_batch(model.qwen_processor, batch["img_pil"], device, visual_dtype)
            ts_inputs = _qwen_process_batch(model.qwen_processor, batch["ts_pil"], device, visual_dtype)

            if img_inputs is None and ts_inputs is None:
                continue

            with autocast_ctx:
                loss, metrics = _compute_losses(
                    model, img_inputs, batch["img_text"], ts_inputs, batch["ts_text"], cfg, device,
                )

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
                msg = (
                    f"step {step:6d} | lr {lr_now:.2e} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    + f" | n_img={len(batch['img_pil'])} n_ts={len(batch['ts_pil'])} | {elapsed:.1f}s"
                )
                logger.info(msg)
                if wandb_run is not None:
                    wandb_run.log({**metrics, "lr": lr_now, "step": step})

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
