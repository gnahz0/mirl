# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage 1 alignment trainer. No Ray, no veRL, no FSDP.

Single-GPU by default. Optionally runs throughput-only DDP (the batch is sharded
across GPUs and gradients are averaged once per optimizer step) -- auto-detected
when launched under ``torchrun`` (WORLD_SIZE>1). See the distributed helpers below.

Usage
-----
    python -m mirl_ext.alignment.trainer \\
        --config mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml
    # multi-GPU:
    torchrun --standalone --nproc_per_node=2 -m mirl_ext.alignment.trainer \\
        --config mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml

The config is a small dict-of-dicts loaded with PyYAML or OmegaConf (we use OmegaConf
since the rest of the repo already depends on it).

What this does
--------------
For each batch of media-bearing samples from the existing Parquet/JSONL schema:

    1. Split by ``data_source`` into ``img`` and ``ts`` branches.
    2. img branch: PIL/video through the Qwen3.5 processor; ts branch: raw signal
       formatted as merger-aware pseudo-image or pseudo-video patches.
    3. ts branch -> trainable VE -> proj head; labels -> frozen SigLIP2 text -> proj head;
       symmetric InfoNCE between the two (``loss/ts_text``).
    4. img branch -> trainable VE AND frozen reference VE; cosine distance between normalized
       raw features (``loss/distill_img``) so image understanding is preserved.
    5. Sum with ``loss_weights``, backprop.

TODOs marked inline for the Stage 2 / veRL handoff.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AlignmentDataset, collate_alignment
from .losses import distill_cosine, info_nce_symmetric
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")


def _setup_logging(level_name: str = "INFO") -> None:
    """Force-install our log handler so other imports don't silently win.

    Also unbuffer stdout/stderr so progress shows up immediately in tmux. Python's
    default line-buffered behavior is fine for TTYs, but Cursor's terminal-mirror
    captures only what's flushed.
    """
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy third-party loggers that emit one INFO line per decoded video,
    # which otherwise bury the per-step training metrics in the terminal.
    for noisy in ("qwen_vl_utils", "qwen_vl_utils.vision_process", "torchcodec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass


def _banner(title: str) -> None:
    """Phase banner -- single flushed line."""
    logger.info("=== %s ===", title)


def _log_param_summary(model) -> None:
    """Log a compact trainable / frozen parameter breakdown so the user can sanity-check
    that the reference VE and SigLIP2 label encoder are both frozen."""

    def _fmt(n: int) -> str:
        return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"

    def _cnt(m):
        return (
            sum(p.numel() for p in m.parameters()),
            sum(p.numel() for p in m.parameters() if p.requires_grad),
        )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Param summary  TOTAL=%s  trainable=%s  (%.2f%%)",
                _fmt(total), _fmt(trainable), 100.0 * trainable / max(1, total))
    for name in (
        "trainable_visual",
        "frozen_visual",
        "label_text_model",
        "proj_visual", "proj_text",
    ):
        if not hasattr(model, name):
            continue
        sub = getattr(model, name)
        t, tr = _cnt(sub)
        tag = "TRAIN" if tr > 0 else "FROZEN"
        logger.info("  %-22s  %-6s  total=%-8s  trainable=%s", name, tag, _fmt(t), _fmt(tr))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "float32": torch.float32,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16}[name.lower()]


# ----------------------------------------------------------------------------
# Distributed (throughput-only DDP: shard the batch across GPUs)
# ----------------------------------------------------------------------------
# This is plain data-parallelism done by hand -- no torch DDP wrapper, no model
# forward refactor. Each rank holds a full model replica and trains on a disjoint
# shard of the data (via DistributedSampler); we average gradients across ranks
# once per optimizer step so every replica takes an identical update and the
# weights stay in lockstep. The contrastive loss still uses only each rank's LOCAL
# negatives (this buys throughput, not a bigger negative pool). Auto-activates when
# launched under torchrun (WORLD_SIZE>1); a plain ``python -m`` run is single-GPU.


def _init_distributed() -> tuple[int, int, int]:
    """Read torchrun env vars and init the NCCL process group if WORLD_SIZE>1.

    Returns ``(rank, local_rank, world_size)``. ``world_size==1`` means a normal
    single-process run -- no collectives are ever issued in that case.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # Passing device_id lets NCCL bind the rank to its GPU up front and
            # avoids the "No device id provided ... barrier" warning at teardown.
            dist.init_process_group(
                backend="nccl", device_id=torch.device("cuda", local_rank)
            )
    return rank, local_rank, world_size


def _allreduce_grad_average(params: list[torch.nn.Parameter], world_size: int) -> None:
    """In-place SUM-allreduce then divide by world_size -> mean gradient on every
    rank. Params with no grad this step (e.g. projection heads on a rank whose
    micro-batch had no ts samples) are given a zero grad first so every rank issues
    the exact same sequence of collectives (otherwise NCCL deadlocks)."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _sync_skip(local_bad: bool, device: torch.device, world_size: int) -> bool:
    """Agree across ranks on whether to skip this micro-batch. Returns True on EVERY
    rank if ANY rank's batch was empty / produced a non-finite loss, so all ranks
    skip together and never diverge in their collective participation."""
    if world_size <= 1:
        return local_bad
    flag = torch.tensor([1.0 if local_bad else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() > 0


def _wandb_has_credentials() -> bool:
    """True if wandb can authenticate -- WANDB_API_KEY env var, ~/.netrc, or
    XDG_CONFIG_HOME/wandb/settings written by ``wandb login``."""
    if "WANDB_API_KEY" in os.environ:
        return True
    for path in (
        os.path.expanduser("~/.netrc"),
        os.path.expanduser("~/.config/wandb/settings"),
    ):
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    if "api.wandb.ai" in f.read():
                        return True
        except OSError:
            pass
    return False


def _maybe_init_wandb(cfg: DictConfig):
    """Initialize a W&B run if enabled in config. Returns the run handle or ``None``.

    Robust to:
        * ``wandb`` not installed -> warn and continue without it.
        * No API key on the machine (env var *or* ~/.netrc) -> falls back to ``WANDB_MODE=offline``.
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

    # Explicit cfg mode wins; otherwise auto-detect credentials.
    if wcfg.get("mode"):
        os.environ.setdefault("WANDB_MODE", str(wcfg.get("mode")))
    elif not _wandb_has_credentials() and os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
        logger.warning(
            "No wandb credentials found (WANDB_API_KEY env var nor ~/.netrc); "
            "defaulting to WANDB_MODE=offline. Run `wandb login` (or set WANDB_API_KEY) to log online."
        )
        os.environ["WANDB_MODE"] = "offline"

    try:
        # console="off" prevents wandb from intercepting stdout/stderr -- otherwise
        # logger output gets buffered and only shown at run finish (very bad UX in tmux).
        run = wandb.init(
            project=str(wcfg.get("project", "mirl-alignment")),
            name=str(wcfg.get("name", "stage1")),
            entity=str(wcfg.get("entity")) if wcfg.get("entity") else None,
            tags=list(wcfg.get("tags", []) or []),
            notes=str(wcfg.get("notes", "")) if wcfg.get("notes") else None,
            group=str(wcfg.get("group")) if wcfg.get("group") else None,
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(console="off"),
        )
        logger.info("W&B run initialized: %s (mode=%s)",
                    run.url if hasattr(run, "url") else "?", os.environ.get("WANDB_MODE", "online"))
        return run
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb init failed (%s); continuing without it", e)
        return None


def _process_images(processor, pil_list, device, dtype):
    """Run the Qwen3.5 processor on a list of PIL images."""
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
    """Run the Qwen3.5 processor on a list of ``(video_tensor, video_metadata)`` pairs.

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


def _n_ts(batch: dict) -> int:
    """Number of raw-signal samples in a collated batch (ts branch)."""
    sig = batch.get("ts_signal")
    return 0 if not sig else len(sig)


@torch.no_grad()
def _classification_metrics(z: torch.Tensor, z_text: torch.Tensor, texts: list) -> Optional[dict]:
    """Zero-shot-style classification metrics for one modality branch.

    The text side IS the ground-truth label, so we treat alignment as classifying
    each anchor (image/ts embedding) against the set of *unique* label texts present
    in the batch. ``z`` / ``z_text`` are already L2-normalized, so cosine sim is a dot
    product. Label-aware: predicting any text that shares the anchor's label counts as
    correct.

    Returns a dict with:
      * "acc"  -- top-1 accuracy over the unique label prototypes.
      * "f1"   -- macro-F1 over present classes.
      * "gap"  -- mean(same-label cosine sim) - mean(diff-label cosine sim). This is
                  the cleanest "is the space actually separating?" signal and is robust
                  to the contrastive-loss floor (loss can look flat while the gap grows
                  or, as with the InfoNCE false-negative problem, stays ~0).
    Returns None if there aren't >=2 samples and >=2 distinct labels to score.
    """
    if z is None or z_text is None or z.shape[0] < 2:
        return None
    label_to_idx: dict = {}
    proto_rows: list = []          # row of z_text to use as each class prototype
    true_ids: list = []
    for i, t in enumerate(texts):
        if t not in label_to_idx:
            label_to_idx[t] = len(label_to_idx)
            proto_rows.append(i)
        true_ids.append(label_to_idx[t])
    num_classes = len(label_to_idx)
    if num_classes < 2:
        return None  # only one label in batch -> accuracy trivially 1, uninformative

    true = torch.tensor(true_ids, device=z.device)               # (B,)
    protos = z_text[proto_rows]                                  # (U, D)
    pred = (z @ protos.t()).argmax(dim=1)                        # (B,)

    acc = (pred == true).float().mean().item()
    f1_sum = 0.0
    for c in range(num_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        denom = 2 * tp + fp + fn
        f1_sum += (2 * tp / denom) if denom > 0 else 0.0

    # Cross-modal anchor<->text cosine sim gap: pos pairs share a label, neg don't.
    sims = z @ z_text.t()                                        # (B, B)
    pos_mask = true.unsqueeze(0) == true.unsqueeze(1)            # (B, B)
    pos_sim = sims[pos_mask].mean().item()
    neg = sims[~pos_mask]
    neg_sim = neg.mean().item() if neg.numel() > 0 else 0.0
    return {"acc": acc, "f1": f1_sum / num_classes, "gap": pos_sim - neg_sim}


def _sanitize_features(
    feat: Optional[torch.Tensor],
    feat_ref: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Zero out any NaN/Inf rows while keeping the batch shape constant.

    The Qwen3.5 ViT can occasionally emit a non-finite row on a degenerate input.
    We replace bad rows with zeros (which become a harmless constant unit vector
    after projection + L2-normalize) rather than dropping them, so the batch
    dimension stays aligned with the text/labels.
    """
    if feat is None or feat.numel() == 0:
        return feat, feat_ref
    bad = (~torch.isfinite(feat)).any(dim=-1)
    if feat_ref is not None and feat_ref.numel() > 0:
        bad = bad | (~torch.isfinite(feat_ref)).any(dim=-1)
    if not bool(bad.any()):
        return feat, feat_ref
    good_mask = (~bad).to(feat.dtype).unsqueeze(-1)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) * good_mask
    if feat_ref is not None and feat_ref.numel() > 0:
        feat_ref = torch.nan_to_num(feat_ref, nan=0.0, posinf=0.0, neginf=0.0) * good_mask
    return feat, feat_ref


@torch.no_grad()
def _run_validation(
    model: MultimodalAlignmentModel,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
    amp_dtype: torch.dtype,
    n_batches: int,
) -> dict:
    """Evaluate over ``n_batches`` batches of the val loader and return mean losses.

    Runs in eval mode with grads off (saves memory + time). The same
    contrastive + distillation losses are computed as in training, just not
    used to update weights.
    """
    was_training = model.training
    model.eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    # Track per-bucket totals across all val batches so users see what the
    # validation distribution actually looked like.
    bucket_totals = {"n/img_image": 0, "n/img_video": 0, "n/ts_signal": 0}
    n_seen = 0
    for batch in val_loader:
        n_img = len(batch["img_image_pil"]) + len(batch["img_video"])
        n_ts = _n_ts(batch)
        if n_img == 0 and n_ts == 0:
            continue
        bucket_totals["n/img_image"] += len(batch["img_image_pil"])
        bucket_totals["n/img_video"] += len(batch["img_video"])
        bucket_totals["n/ts_signal"] += n_ts
        with autocast_ctx:
            _, metrics = _compute_losses(model, batch, cfg, device, visual_dtype)
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if v == 0.0 and k.startswith("loss/"):
                # Skip pre-populated loss placeholders; don't dilute val averages
                # with branches that weren't computed on this batch. (acc/f1 are
                # only present when computed, so a genuine 0.0 there is kept.)
                continue
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
        n_seen += 1
        if n_seen >= n_batches:
            break

    if was_training:
        model.train()
        model.frozen_visual.eval()
        model.label_text_model.eval()

    out = {f"val/{k}": (sums[k] / counts[k]) for k in sums}
    # Add per-bucket sample totals so val gives the same distribution view as train.
    for k, v in bucket_totals.items():
        out[f"val/{k}"] = float(v)
    out["val/n_batches"] = float(n_seen)
    return out


def _sanitize_grads(model: torch.nn.Module) -> int:
    """Replace NaN/Inf in parameter gradients with zeros (post-backward, pre-step).

    Even though we zero NaN features pre-projection, the visual-encoder backward
    pass can still produce NaN gradients on its own parameters when its forward
    had internal NaN intermediates. We scrub those before the optimizer step so
    the weights themselves never become NaN.
    Returns the number of params whose gradients were touched.
    """
    n_touched = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            n_touched += 1
    return n_touched


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Two losses:

    * ``ts_text``     -- InfoNCE: trainable VE(signal pixels) <-> SigLIP2(label text),
                         both projected to the shared contrastive dim. Teaches the VE
                         the new modality.
    * ``distill_img`` -- cosine distance on normalized raw VE features over
                         images + video frames. No projection heads -- anchors the exact
                         features the LM tower will consume in Stage 2, so image
                         understanding is preserved while the VE learns signals.
    """
    # Pre-populate loss keys with 0.0 so W&B charts have a point on every step
    # (a branch may be empty in a given batch).
    metrics: dict[str, float] = {
        "loss/ts_text": 0.0,
        "loss/distill_img": 0.0,
    }

    # img branch: normal images + video frames concatenated, in that order.
    feat_img, feat_ref_img = _encode_branch(
        model, model.qwen_processor,
        batch["img_image_pil"], batch["img_video"], device, visual_dtype,
    )
    # ts branch: raw signals formatted as pseudo-images/videos -> same VE as the img branch.
    # Only the trainable VE runs here (no frozen pass: the frozen VE has never seen
    # signal pseudo-images, so it is not a meaningful teacher for them).
    feat_ts = None
    ts_signal = batch.get("ts_signal")
    if ts_signal:
        feat_ts = model.encode_ts_trainable(
            ts_signal, batch.get("ts_format") or [], device=device, dtype=visual_dtype
        )

    ts_text = list(batch["ts_signal_text"])

    # Zero out any non-finite ViT rows (keeps the batch dim aligned with the text).
    feat_img, feat_ref_img = _sanitize_features(feat_img, feat_ref_img)
    feat_ts, _ = _sanitize_features(feat_ts, None)

    total = torch.zeros((), device=device, dtype=torch.float32)
    w = cfg.loss_weights

    # ---- ts_text: symmetric InfoNCE (diagonal positives) in the projected space ----
    z_ts = model.project(model.proj_visual, feat_ts) if feat_ts is not None else None
    z_text_ts = None
    if z_ts is not None and ts_text:
        z_text_ts = model.project(model.proj_text, model.encode_text(ts_text, device=device).float())
    if z_ts is not None and z_text_ts is not None and z_ts.shape[0] > 1:
        # Duplicate-label-aware positives: signals sharing the SAME text (e.g. multiple
        # "Normal" ECGs, or repeated smellnet labels) must not be treated as negatives of
        # each other. Build a symmetric (B,B) same-text mask (diagonal always True).
        pos_mask = torch.tensor(
            [[t1 == t2 for t2 in ts_text] for t1 in ts_text],
            device=device, dtype=torch.bool,
        )
        l_ts = info_nce_symmetric(z_ts, z_text_ts, model.log_logit_scale, pos_mask=pos_mask)
        total = total + float(w.ts_text) * l_ts
        metrics["loss/ts_text"] = l_ts.detach().item()
        cm = _classification_metrics(z_ts, z_text_ts, ts_text)
        if cm is not None:
            metrics["acc/ts_text"] = cm["acc"]
            metrics["f1/ts_text"] = cm["f1"]
            metrics["gap/ts_text"] = cm["gap"]

    # ---- distill_img: raw-feature cosine distance, trainable vs frozen VE ----
    if feat_img is not None and feat_ref_img is not None and feat_img.shape[0] > 0:
        l_img = distill_cosine(
            model._norm(feat_img.float()), model._norm(feat_ref_img.float()),
        )
        total = total + float(w.distill_img) * l_img
        metrics["loss/distill_img"] = l_img.detach().item()

    metrics["loss/total"] = total.detach().item()
    metrics["logit_scale"] = model.log_logit_scale.detach().exp().item()
    return total, metrics


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    _setup_logging(cfg.get("log_level", "INFO"))

    # Distributed (throughput DDP). Single-process when not launched via torchrun.
    rank, local_rank, world_size = _init_distributed()
    is_dist = world_size > 1
    is_main = rank == 0
    if not is_main:
        # Silence duplicate INFO spam from non-zero ranks; keep warnings/errors.
        logging.getLogger().setLevel(logging.WARNING)

    amp_dtype = _resolve_dtype(cfg.train.amp_dtype)
    visual_dtype = _resolve_dtype(cfg.model.trainable_visual_dtype)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner("Stage 1 multimodal alignment trainer")
    if is_dist:
        logger.info("DDP: rank %d/%d (local_rank %d) -- batch sharded across %d GPUs",
                    rank, world_size, local_rank, world_size)
    logger.info("device: %s  amp_dtype: %s  visual_dtype: %s", device, amp_dtype, visual_dtype)
    logger.info("output dir: %s", out_dir)
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GiB total)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))

    # ---- Data ----
    _banner("Building dataset")
    t_data = time.time()
    train_ds = AlignmentDataset(
        data_files=list(cfg.data.train_files),
        ts_data_sources=list(cfg.data.ts_data_sources),
        text_for_label=cfg.data.text_for_label,
        max_samples=int(cfg.data.get("max_train_samples", -1)),
        balanced_sampling_key=cfg.data.get("balanced_sampling_key"),
        seed=int(cfg.train.get("seed", 42)),
        enable_videos=bool(cfg.data.get("enable_videos", True)),
        max_video_frames=cfg.data.get("max_video_frames"),
        image_patch_size=int(cfg.data.get("image_patch_size", 14)),
        video_load_timeout=int(cfg.data.get("video_load_timeout", 30)),
        video_suppress_stderr=bool(cfg.data.get("video_suppress_stderr", True)),
        data_source_filter=list(cfg.data.get("data_source_filter") or []) or None,
        ts_in_channels=cfg.data.get("ts_in_channels"),
        ts_seq_len=cfg.data.get("ts_seq_len"),
        ts_oversample=int(cfg.data.get("ts_oversample", 1)),
        ts_pt_target_len=cfg.data.get("ts_pt_target_len"),
        tactile_max_frames=cfg.data.get("tactile_max_frames"),
        include_all_ts=bool(cfg.data.get("include_all_ts", False)),
        max_img_samples=int(cfg.data.get("max_img_samples", -1)),
    )
    logger.info("train dataset: %d media-bearing samples (loaded in %.1fs)",
                len(train_ds), time.time() - t_data)

    # CRITICAL: fork() after CUDA init is a textbook PyTorch deadlock. The trainer
    # touches CUDA earlier (model.to(device)), so any DataLoader worker forked from
    # this process will inherit a poisoned CUDA context and hang forever.
    #
    # Two safe options:
    #   * num_workers=0  -> all preprocessing in main process (simple, slow-ish)
    #   * spawn context  -> workers re-import everything from scratch (~5-10s startup,
    #     but parallel preprocessing afterwards)
    #
    # We default to num_workers=2 with spawn. User can override via config.
    num_workers = int(cfg.train.get("num_workers", 2))
    # Under DDP each rank gets a disjoint shard via DistributedSampler (the sampler
    # owns shuffling, so DataLoader shuffle must be False). Single-GPU keeps shuffle.
    train_sampler = None
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=int(cfg.train.get("seed", 42)),
        )
    dl_kwargs = dict(
        batch_size=int(cfg.train.batch_size),
        num_workers=num_workers,
        collate_fn=collate_alignment,
        pin_memory=True,
        drop_last=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    if num_workers > 0:
        import multiprocessing as _mp
        mp_ctx = str(cfg.train.get("mp_context", "spawn"))
        dl_kwargs["multiprocessing_context"] = _mp.get_context(mp_ctx)
        dl_kwargs["persistent_workers"] = True
        prefetch_factor = int(cfg.train.get("prefetch_factor", 2))
        dl_kwargs["prefetch_factor"] = prefetch_factor
        logger.info("DataLoader: %d workers, mp_context=%s, persistent=True, prefetch_factor=%d",
                    num_workers, mp_ctx, prefetch_factor)
    else:
        logger.info("DataLoader: num_workers=0 (single-process preprocessing)")
    train_loader = DataLoader(train_ds, **dl_kwargs)

    # ---- Validation dataset (small, fixed; eval'd periodically) ----
    val_loader: Optional[DataLoader] = None
    val_files = list(cfg.data.get("val_files") or [])
    val_every = int(cfg.train.get("val_every", 0))   # 0 disables val
    val_batches = int(cfg.train.get("val_batches", 32))  # how many batches to eval each time
    if val_files and val_every > 0:
        t_val = time.time()
        val_ds = AlignmentDataset(
            data_files=val_files,
            ts_data_sources=list(cfg.data.ts_data_sources),
            text_for_label=cfg.data.text_for_label,
            max_samples=int(cfg.data.get("max_val_samples", val_batches * int(cfg.train.batch_size) * 4)),
            balanced_sampling_key=cfg.data.get("balanced_sampling_key"),
            seed=int(cfg.train.get("seed", 42)),
            enable_videos=bool(cfg.data.get("enable_videos", True)),
            max_video_frames=cfg.data.get("max_video_frames"),
            image_patch_size=int(cfg.data.get("image_patch_size", 14)),
            video_load_timeout=int(cfg.data.get("video_load_timeout", 30)),
            video_suppress_stderr=bool(cfg.data.get("video_suppress_stderr", True)),
            data_source_filter=list(cfg.data.get("data_source_filter") or []) or None,
            ts_in_channels=cfg.data.get("ts_in_channels"),
            ts_seq_len=cfg.data.get("ts_seq_len"),
            # Same ts oversampling as training: smellnet is ~1% of the val pool, and
            # the ts InfoNCE/acc metrics only compute when >=2 ts samples share a val
            # batch -- without this they fire too rarely to give a stable val curve.
            ts_oversample=int(cfg.data.get("ts_oversample", 1)),
            ts_pt_target_len=cfg.data.get("ts_pt_target_len"),
            tactile_max_frames=cfg.data.get("tactile_max_frames"),
        )
        logger.info("val dataset: %d samples (loaded in %.1fs)",
                    len(val_ds), time.time() - t_val)
        # Validation is forward-only (no stored activations), so it can run a much
        # larger batch than training -- more samples per eval at the same wall time.
        val_bs = int(cfg.train.get("val_batch_size", 4 * int(cfg.train.batch_size)))
        logger.info("val batch_size=%d (%d batches per eval -> ~%d samples)",
                    val_bs, val_batches, val_bs * val_batches)
        val_dl_kwargs = dict(
            batch_size=val_bs,
            num_workers=0,  # tiny + only used during eval, no point in workers
            collate_fn=collate_alignment,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        val_loader = DataLoader(val_ds, **val_dl_kwargs)
    elif val_files:
        logger.info("val_files configured but train.val_every=0; skipping validation.")

    # ---- Model ----
    _banner("Building model (exact Qwen3.5 VE + frozen ref + SigLIP2 label text)")
    t_model = time.time()
    model = MultimodalAlignmentModel(
        qwen35_path=str(cfg.model.qwen35_path),
        siglip2_text_path=str(cfg.model.siglip2_text_path),
        shared_dim=int(cfg.projection.shared_dim),
        proj_hidden_dim=int(cfg.projection.hidden_dim),
        proj_dropout=float(cfg.projection.get("dropout", 0.0)),
        visual_dtype=visual_dtype,
        attn_impl=str(cfg.model.get("attn_impl", "sdpa")),
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        ts_representation=str(cfg.model.get("ts_representation", "hybrid")),
    ).to(device)
    logger.info("model build done in %.1fs", time.time() - t_model)

    # Master weights in fp32, activations in bf16 (standard mixed-precision recipe).
    #
    # Why: previously the trainable visual encoder lived in bfloat16 end-to-end --
    # params, gradients, AND AdamW's m/v moments all in bf16. bf16 has only ~7 bits
    # of mantissa, so for parameters with small gradients (LayerNorm gains, MLP
    # biases) the per-step Adam update is smaller than the LSB of the bf16 weight,
    # the cast-back-to-bf16 rounds to zero, and the weight is bit-exactly stuck.
    # In the Stage-1 run this kept ~25% of visual params (norm.weight, fc1.bias,
    # etc.) frozen for the entire run.
    #
    # The fix: keep ALL trainable parameters as fp32 master weights. The autocast
    # context below still does bf16 forward/backward for matmuls + convs (so we
    # keep the speed and activation-memory benefits of mixed precision), but Adam
    # accumulates updates in fp32 and the weights themselves move at fp32
    # resolution. Frozen VE + frozen SigLIP2 stay bf16 (no optimizer state for them).
    n_upcast = 0
    n_upcast_params = 0
    for p_name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n_upcast += 1
            n_upcast_params += p.numel()
    if n_upcast > 0:
        logger.info(
            "Upcast %d trainable tensors (%.1fM params) to fp32 master weights "
            "(bf16 forward via autocast).",
            n_upcast, n_upcast_params / 1e6,
        )

    # Optional warm start: load trainable weights from a previous checkpoint and
    # continue training (fresh optimizer + LR schedule -- see _load_checkpoint).
    resume_from = cfg.train.get("resume_from")
    if resume_from:
        _load_checkpoint(model, Path(str(resume_from)))

    # Print trainable / frozen parameter summary so the user can sanity-check freezing.
    if is_main:
        _log_param_summary(model)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "GPU memory after model load: used %.2f GiB / %.2f GiB (free %.2f GiB)",
            (total - free) / (1024 ** 3), total / (1024 ** 3), free / (1024 ** 3),
        )

    # ---- Optim ----
    # The from-scratch projection heads train at a higher LR than the pretrained
    # Qwen3.5 ViT; the cosine schedule scales both groups proportionally from their
    # own base LR (LambdaLR multiplies each group's initial_lr by the same lambda).
    head_lr = float(cfg.train.get("head_lr", cfg.train.lr))
    optim_groups = model.trainable_parameter_groups(
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay), head_lr=head_lr,
    )
    logger.info("optim LR tiers: ViT=%.2e  heads=%.2e", float(cfg.train.lr), head_lr)
    optimizer = torch.optim.AdamW(
        optim_groups, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        betas=(0.9, 0.95), eps=1e-8,
    )

    total_steps = int(cfg.train.total_steps)  # NB: counted in OPTIMIZER STEPS (not micro-batches)
    warmup = int(cfg.train.warmup_steps)

    def lr_lambda(opt_step: int) -> float:
        # ``opt_step`` is the number of completed optimizer.step() calls. The
        # LambdaLR scheduler advances it by 1 each time ``scheduler.step()`` is
        # called, which we only do once per ``grad_accum`` micro-batches. So the
        # warmup + cosine schedule is measured in optimizer steps, matching the
        # ``total_steps`` config field below.
        if opt_step < warmup:
            return float(opt_step + 1) / max(1, warmup)
        progress = (opt_step - warmup) / max(1, total_steps - warmup)
        import math as _m
        return 0.5 * (1.0 + _m.cos(_m.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- Optional W&B (rank 0 only) ----
    wandb_run = _maybe_init_wandb(cfg) if is_main else None

    # ---- Train ----
    model.train()
    model.frozen_visual.eval()
    model.label_text_model.eval()

    # Step counters:
    #   ``micro_step`` -- micro-batches processed (used only to decide when to
    #                     fire optimizer.step(): every ``grad_accum`` micro-batches).
    #   ``opt_step``   -- completed optimizer.step() calls. THIS is the canonical
    #                     "step" counter for the LR schedule, pbar, log_every,
    #                     ckpt_every, and the stop condition (total_steps).
    micro_step = 0
    opt_step = 0
    grad_accum = max(1, int(cfg.train.get("grad_accum_steps", 1)))
    log_every = int(cfg.train.get("log_every", 10))
    ckpt_every = int(cfg.train.get("ckpt_every", 500))
    # Best-checkpoint tracking: keep the highest-generalization snapshot (by
    # held-out val/acc/ts_text) so late overfitting on the ~1.9k repeated signals
    # can't cost us the good weights. ``best/`` is what Stage 2 should consume.
    best_metric_key = str(cfg.train.get("best_metric", "val/acc/ts_text"))
    best_metric_val = float("-inf")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    _banner(
        f"Training: {total_steps} optimizer-steps "
        f"(micro-batch={cfg.train.batch_size}, grad_accum={grad_accum}, "
        f"effective batch={int(cfg.train.batch_size) * grad_accum})"
    )
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    # Stable list of trainable params for the manual gradient all-reduce (same
    # order on every rank since the model is constructed identically).
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Progress bar / stdout logging live on rank 0 only (other ranks stay quiet).
    pbar = tqdm(
        total=total_steps,
        desc="stage1",
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stdout,
    ) if is_main else None
    # Accumulated per-bucket sample counts across the current grad_accum window.
    # Reset to 0 every opt step. This way the W&B "n/*" series shows the TOTAL
    # samples that contributed to each optimizer update, not just the last
    # micro-batch in the window.
    accum_counts = {"n/img_image": 0, "n/img_video": 0, "n/ts_signal": 0}

    done = False
    epoch = 0
    while not done:
        # DistributedSampler needs a fresh epoch each pass for proper reshuffling.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            n_img = len(batch["img_image_pil"]) + len(batch["img_video"])
            n_ts = _n_ts(batch)

            with autocast_ctx:
                loss, metrics = _compute_losses(model, batch, cfg, device, visual_dtype)

            # Decide (in lockstep across ranks) whether to skip this micro-batch:
            # an empty batch yields a leaf zero loss (no grad_fn); a degenerate input
            # can yield a non-finite loss. Under DDP every rank must skip together or
            # the per-step gradient all-reduce deadlocks.
            local_bad = (
                (n_img == 0 and n_ts == 0)
                or not loss.requires_grad
                or not torch.isfinite(loss)
            )
            if _sync_skip(local_bad, device, world_size):
                if local_bad and loss.requires_grad and not torch.isfinite(loss):
                    logger.warning("non-finite loss near opt_step %d; skipping micro-batch", opt_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            # Tally this micro-batch's contribution to the current accumulation window.
            accum_counts["n/img_image"] += len(batch["img_image_pil"])
            accum_counts["n/img_video"] += len(batch["img_video"])
            accum_counts["n/ts_signal"] += n_ts

            (loss / grad_accum).backward()
            micro_step += 1
            took_opt_step = False

            if micro_step % grad_accum == 0:
                # Scrub any NaN/Inf grads so one bad input can't corrupt the weights.
                _sanitize_grads(model)
                # Average gradients across ranks BEFORE clipping so every replica
                # clips on the same global grad norm and takes an identical step.
                if is_dist:
                    _allreduce_grad_average(trainable_params, world_size)

                if float(cfg.train.get("grad_clip", 0.0)) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(cfg.train.grad_clip),
                    )
                    metrics["grad_norm"] = float(grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                took_opt_step = True

            # Use accumulated counts for logging when we're about to take an
            # optimizer step (so the n/* numbers reflect the FULL effective batch).
            # On non-opt-step micro-batches there's no log anyway.
            counts = dict(accum_counts) if took_opt_step else {
                "n/img_image": len(batch["img_image_pil"]),
                "n/img_video": len(batch["img_video"]),
                "n/ts_signal": _n_ts(batch),
            }
            lr_now = optimizer.param_groups[0]["lr"]            # ViT base (group 0)
            lr_head = optimizer.param_groups[-1]["lr"]          # heads (last group)

            # Update pbar / log / ckpt only on a completed optimizer step. That way
            # the visible "step" matches the LR schedule and config.total_steps.
            # Logged metrics are rank-0-local (we don't all-reduce metrics).
            if took_opt_step:
                if is_main:
                    pbar.set_postfix({
                        "loss": f"{metrics.get('loss/total', float('nan')):.3f}",
                        "lr": f"{lr_now:.1e}",
                        "img": counts["n/img_image"] + counts["n/img_video"],
                        "ts": counts["n/ts_signal"],
                    }, refresh=False)
                    pbar.update(1)

                    # W&B: log every opt step so every metric has a value at every step.
                    # Terminal log is kept gated by ``log_every`` so we don't spam stdout.
                    if wandb_run is not None:
                        wandb_run.log(
                            {**metrics, **counts, "lr": lr_now, "lr_head": lr_head, "opt_step": opt_step},
                            step=opt_step,
                        )

                    if opt_step % log_every == 0:
                        elapsed = time.time() - t0
                        # Peak allocator high-water mark since last log; reset after reading
                        # so each line reports the peak of the interval (cheap, ~free).
                        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
                        torch.cuda.reset_peak_memory_stats()
                        msg = (
                            f"opt_step {opt_step:6d} | lr {lr_now:.2e} | "
                            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("loss") or k == "grad_norm")
                            + " | "
                            + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("acc/") or k.startswith("f1/") or k.startswith("gap/"))
                            + " | "
                            + " ".join(f"{k}={v}" for k, v in counts.items())
                            + f" | peak {peak_gib:.1f}GiB | {elapsed:.1f}s"
                        )
                        # tqdm.write keeps the progress bar intact while emitting a log line.
                        tqdm.write(msg, file=sys.stdout)

                    if opt_step > 0 and opt_step % ckpt_every == 0:
                        # Keep only the most recent checkpoint: overwrite a single
                        # ``latest/`` dir instead of accumulating ``step_N/`` dirs.
                        _save_checkpoint(model, out_dir / "latest", cfg, step=opt_step)

                # Periodic validation -- all ranks reach this together (opt_step is
                # identical across ranks). Only rank 0 evaluates; the others wait at
                # the barrier so no one races ahead into the next grad all-reduce.
                if val_loader is not None and opt_step > 0 and opt_step % val_every == 0:
                    if is_dist:
                        dist.barrier()
                    if is_main:
                        val_metrics = _run_validation(
                            model, val_loader, cfg, device, visual_dtype, amp_dtype,
                            n_batches=val_batches,
                        )
                        tqdm.write(
                            "VAL @ opt_step "
                            f"{opt_step:6d} | "
                            + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                            file=sys.stdout,
                        )
                        if wandb_run is not None:
                            wandb_run.log({**val_metrics, "opt_step": opt_step}, step=opt_step)
                        # Save the best-generalizing checkpoint (higher acc = better).
                        cur = val_metrics.get(best_metric_key)
                        if cur is not None and cur > best_metric_val:
                            best_metric_val = cur
                            _save_checkpoint(model, out_dir / "best", cfg, step=opt_step)
                            tqdm.write(
                                f"  ^ new best {best_metric_key}={cur:.4f} -> saved best/ (step {opt_step})",
                                file=sys.stdout,
                            )
                            if wandb_run is not None:
                                wandb_run.log({"best/metric": cur, "best/step": opt_step}, step=opt_step)
                    if is_dist:
                        dist.barrier()

                opt_step += 1
                # Reset accumulation window for the next optimizer step.
                accum_counts = {"n/img_image": 0, "n/img_video": 0, "n/ts_signal": 0}
                if opt_step >= total_steps:
                    done = True
                    break

    if pbar is not None:
        pbar.close()
    if is_main:
        _save_checkpoint(model, out_dir / "final", cfg, step=opt_step)
    if wandb_run is not None:
        wandb_run.finish()
    if is_dist:
        dist.barrier()  # ensure rank 0 finishes writing before anyone exits
        dist.destroy_process_group()
    if is_main and best_metric_val > float("-inf"):
        logger.info("best %s=%.4f saved to %s (prefer this over final/ for Stage 2)",
                    best_metric_key, best_metric_val, out_dir / "best")
    _banner(f"Stage 1 alignment complete. final={out_dir / 'final'} best={out_dir / 'best'}")


def _save_checkpoint(
    model: MultimodalAlignmentModel,
    path: Path,
    cfg: DictConfig,
    step: Optional[int] = None,
) -> None:
    """Save the trainable VE state dict and projection heads.
    The frozen reference VE and SigLIP2 text encoder are not saved (recoverable from HF).

    Writes to a sibling ``<name>.tmp`` dir first, then atomically renames over the
    target. This makes overwriting a single ``latest/`` dir crash-safe: a kill
    mid-save can't leave a half-written checkpoint at ``path``.

    TODO(stage2): also write a tiny ``visual_only.safetensors`` ready to be dropped into
    a Qwen3.5 HF checkpoint dir's ``model.visual.*`` keys for the veRL handoff.
    """
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trainable_visual": model.trainable_visual.state_dict(),
        "proj_visual": model.proj_visual.state_dict(),
        "proj_text": model.proj_text.state_dict(),
        "log_logit_scale": model.log_logit_scale.detach().cpu(),
        "step": step,
    }, tmp / "alignment_state.pt")
    OmegaConf.save(cfg, tmp / "config.yaml")
    # Atomic swap: remove any existing checkpoint then rename temp into place.
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    logger.info("checkpoint saved to %s (step=%s)", path, step)


def _load_checkpoint(model: MultimodalAlignmentModel, path: Path) -> Optional[int]:
    """Warm-start the trainable modules from a checkpoint saved by ``_save_checkpoint``.

    Loads the trainable VE, both projection heads, and the logit scale. The frozen
    reference VE + SigLIP2 text stay at their checkpoint init (they are never trained).
    NOTE: optimizer/scheduler state is NOT restored (we only persist weights), so this
    is a *warm start* -- training resumes with a fresh optimizer and a fresh LR
    schedule, not an exact mid-run resume.
    """
    ckpt_file = (path / "alignment_state.pt") if path.is_dir() else path
    state = torch.load(ckpt_file, map_location="cpu")
    model.trainable_visual.load_state_dict(state["trainable_visual"])
    model.proj_visual.load_state_dict(state["proj_visual"])
    model.proj_text.load_state_dict(state["proj_text"])
    with torch.no_grad():
        model.log_logit_scale.copy_(state["log_logit_scale"].to(model.log_logit_scale.device))
    saved_step = state.get("step")
    logger.info("warm-started weights from %s (saved step=%s)", ckpt_file, saved_step)
    return saved_step


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 multimodal alignment trainer")
    p.add_argument("--config", "-c", required=True, type=str,
                   help="Path to YAML config (see mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml)")
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
