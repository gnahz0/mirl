# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 alignment objective, prototype banks, and evaluation."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.nn.functional import all_gather as grad_all_gather
from torch.utils.data import DataLoader

from .losses import distill_cosine, siglip_sigmoid
from .metrics import (
    _TS_FAMILIES,
    _smellnet_gcms_top1_metrics,
    _ts_prediction_metrics,
    _validation_metric_groups,
    add_ts_family_counts,
    new_counts,
)
from .model import MultimodalAlignmentModel, temporal_crop
from .smellnet_gcms import SmellNetGCMSBank

logger = logging.getLogger("alignment.trainer")


@dataclass(frozen=True)
class TextPrototypeFamily:
    """Frozen SigLIP2 features and stable label ordering for one TS family."""

    labels: tuple[str, ...]
    raw_features: torch.Tensor


def _log_visual_batch_once(model: MultimodalAlignmentModel, kind: str, grid: torch.Tensor) -> None:
    flag = f"_logged_{kind}_batch"
    if getattr(model, flag, False):
        return
    postmerge = grid.prod(dim=1) // model.vit_merge_size**2
    logger.info(
        "first %s bucket: samples=%d, post-merger tokens total=%d max/sample=%d",
        kind,
        grid.shape[0],
        postmerge.sum().item(),
        postmerge.max().item(),
    )
    setattr(model, flag, True)


def _encode_branch(
    model: MultimodalAlignmentModel,
    images_pil,
    videos,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[int]]:
    """Encode image/video post-merger tokens with student and frozen Qwen towers."""
    if not images_pil and not videos:
        return None, None, []

    train_parts: list[torch.Tensor] = []
    frozen_parts: list[torch.Tensor] = []
    rows_per_sample: list[int] = []

    if images_pil:
        processed = model.qwen_processor(
            images=images_pil,
            text=["<image>"] * len(images_pil),
            return_tensors="pt",
            padding=True,
        )
        pixels = processed["pixel_values"].to(device=device, dtype=dtype)
        grid = processed["image_grid_thw"].to(device=device)
        _log_visual_batch_once(model, "image", grid)
        train_parts.append(model.encode_visual(pixels, grid, pool=False))
        frozen_parts.append(model.encode_visual(pixels, grid, frozen=True, pool=False))
        rows_per_sample.extend((grid[:, 0] * grid[:, 1] * grid[:, 2] // model.vit_merge_size**2).tolist())

    if videos:
        tensors, metadata = zip(*videos, strict=True)
        processed = model.qwen_processor(
            videos=list(tensors),
            text=["<video>"] * len(tensors),
            return_tensors="pt",
            padding=True,
            videos_kwargs={"video_metadata": list(metadata), "do_sample_frames": False},
        )
        pixels = processed["pixel_values_videos"].to(device=device, dtype=dtype)
        grid = processed["video_grid_thw"].to(device=device)
        _log_visual_batch_once(model, "video", grid)
        train_parts.append(model.encode_visual(pixels, grid, pool=False))
        frozen_parts.append(model.encode_visual(pixels, grid, frozen=True, pool=False))
        rows_per_sample.extend((grid[:, 0] * grid[:, 1] * grid[:, 2] // model.vit_merge_size**2).tolist())

    train_tokens = torch.cat(train_parts, dim=0)
    frozen_tokens = torch.cat(frozen_parts, dim=0)
    if sum(rows_per_sample) != train_tokens.shape[0]:
        raise ValueError(f"post-merger token rows {train_tokens.shape[0]} != sample counts {rows_per_sample}")
    return train_tokens, frozen_tokens, rows_per_sample


@torch.no_grad()
def _build_text_prototype_bank(
    model: MultimodalAlignmentModel,
    vocabularies: dict[str, tuple[str, ...]],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, TextPrototypeFamily]:
    """Encode each complete training label vocabulary once with frozen SigLIP2."""
    bank: dict[str, TextPrototypeFamily] = {}
    for family in _TS_FAMILIES:
        labels = tuple(vocabularies.get(family, ()))
        if not labels:
            continue
        if len(set(labels)) != len(labels):
            raise ValueError(f"prototype labels for {family!r} are not unique")
        chunks = [
            model.encode_text(list(labels[start : start + batch_size]), device=device).float()
            for start in range(0, len(labels), batch_size)
        ]
        bank[family] = TextPrototypeFamily(
            labels=labels,
            raw_features=torch.cat(chunks, dim=0).detach(),
        )
        logger.info("prototype bank: family=%s classes=%d", family, len(labels))
    return bank


def _project_text_prototype_bank(
    model: MultimodalAlignmentModel,
    bank: dict[str, TextPrototypeFamily],
    families: set[str],
) -> dict[str, tuple[tuple[str, ...], torch.Tensor]]:
    """Run cached frozen text features through the trainable text projection."""
    return {
        family: (entry.labels, model.project(model.proj_text, entry.raw_features))
        for family, entry in bank.items()
        if family in families
    }


def _project_gcms_bank(
    model: MultimodalAlignmentModel,
    bank: SmellNetGCMSBank,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Project the fixed 50-class GC-MS bank into the normalized shared space."""
    return bank.labels, model.project(model.proj_gcms, bank.features)


def _paired_prototype_siglip_loss(
    left: tuple[tuple[str, ...], torch.Tensor],
    right: tuple[tuple[str, ...], torch.Tensor],
    log_logit_scale: torch.Tensor,
) -> torch.Tensor:
    """SigLIP over two aligned one-prototype-per-class banks."""
    left_labels, left_features = left
    _, right_features = right
    classes = len(left_labels)
    positive_rate = 1.0 / classes
    bias = left_features.new_tensor(math.log(positive_rate / (1.0 - positive_rate)))
    positive = torch.eye(classes, device=left_features.device, dtype=torch.bool)
    return siglip_sigmoid(
        left_features,
        right_features,
        log_logit_scale,
        bias,
        pos_mask=positive,
    )


@torch.no_grad()
def _score_ts_collector(
    model: MultimodalAlignmentModel,
    collector: dict[str, list],
    prototype_bank: dict[str, TextPrototypeFamily],
    gcms_bank: Optional[SmellNetGCMSBank] = None,
    per_class_reports: Optional[dict[str, list[dict[str, object]]]] = None,
) -> dict[str, float]:
    """Score one fixed TS sample set against the current projected prototypes."""
    if not collector["z"]:
        return {}
    z = torch.cat(collector["z"], dim=0)
    labels = collector["labels"]
    families = collector["families"]
    projected_bank = _project_text_prototype_bank(
        model,
        prototype_bank,
        set(families),
    )
    metrics = _ts_prediction_metrics(
        z,
        labels,
        families,
        projected_bank,
        per_class_reports=per_class_reports,
    )
    if gcms_bank is not None:
        text_smell = projected_bank.get("smell")
        if text_smell is None:
            raise ValueError("GC-MS metrics require SmellNet text prototypes")
        metrics.update(
            _smellnet_gcms_top1_metrics(
                z,
                labels,
                families,
                text_smell,
                _project_gcms_bank(model, gcms_bank),
            )
        )
    return metrics


def _sanitize_features(
    feat: Optional[torch.Tensor],
    feat_ref: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Zero non-finite feature rows without changing label alignment."""
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
    prototype_bank: dict[str, TextPrototypeFamily],
    gcms_bank: Optional[SmellNetGCMSBank] = None,
) -> tuple[
    dict[str, float],
    dict[str, list[dict[str, object]]],
]:
    """Evaluate losses and prediction metrics over a fixed validation subset."""
    was_training = model.training
    model.eval()

    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    # Collect embeddings once, then score the fixed validation subset globally.
    ts_eval: dict[str, list] = {"z": [], "labels": [], "families": []}
    # Track per-bucket totals across all val batches so users see what the
    # validation distribution actually looked like.
    bucket_totals = new_counts()
    n_seen = 0
    for batch in val_loader:
        n_img = len(batch["img_image_pil"]) + len(batch["img_video"])
        n_ts = len(batch.get("ts_signal") or [])
        if n_img == 0 and n_ts == 0:
            continue
        bucket_totals["n/img_image"] += len(batch["img_image_pil"])
        bucket_totals["n/img_video"] += len(batch["img_video"])
        bucket_totals["n/ts_signal"] += n_ts
        # Per-family denominators. Without these the per-family accuracy/F1 below are
        # uninterpretable: a family contributing 3 rows per batch produces a noisy
        # number that looks exactly like one measured over 300.
        add_ts_family_counts(bucket_totals, batch)
        with autocast_ctx:
            # world_size=1 DELIBERATELY: validation is rank-0-only, so gathering here
            # would issue collectives no other rank answers and hang until the NCCL
            # watchdog fires. The LOSS remains batch-local (~16 TS rows/batch), but
            # accuracy/F1/gap are replaced below by one calculation over every TS row
            # collected from this fixed validation subset (~519 in production).
            _, metrics = _compute_losses(
                model,
                batch,
                cfg,
                device,
                visual_dtype,
                ts_eval_collector=ts_eval,
                prototype_bank=prototype_bank,
                gcms_bank=gcms_bank,
            )
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith(
                (
                    "accuracy/",
                    "f1_",
                    "gap/",
                    "eff_dim/",
                    "label_coverage/",
                    "class_coverage/",
                    "prediction_coverage/",
                )
            ):
                continue  # replaced by one global, sample-correct calculation below
            # Every key present is a real measurement (no placeholders), so each
            # is averaged over exactly the batches where its branch fired.
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
        n_seen += 1
        if n_seen >= n_batches:
            break

    averaged = {key: (sums[key] / counts[key]) for key in sums}
    per_class_reports: dict[str, list[dict[str, object]]] = {}
    prediction_metrics = _score_ts_collector(
        model,
        ts_eval,
        prototype_bank,
        gcms_bank=gcms_bank,
        per_class_reports=per_class_reports,
    )

    if was_training:
        model.train()
        model.frozen_visual.eval()
        model.label_text_model.eval()

    out = _validation_metric_groups(
        averaged,
        prediction_metrics,
        bucket_totals,
    )
    return out, per_class_reports


def _gather_ts_embeddings(
    z_ts: Optional[torch.Tensor],
    ts_text: list,
    ts_family: list,
    device: torch.device,
    world_size: int,
    shared_dim: int,
):
    """Differentiably gather uneven TS rows and metadata from every rank."""
    empty = z_ts is None
    local_n = 0 if empty else int(z_ts.shape[0])

    # Agree on the padded width. All ranks, unconditionally.
    n_t = torch.tensor([local_n], device=device, dtype=torch.long)
    dist.all_reduce(n_t, op=dist.ReduceOp.MAX)
    max_n = int(n_t.item())
    if max_n == 0:
        return None, None, None

    # Use fp32 everywhere so empty and non-empty ranks agree on collective dtype.
    if empty:
        # Zeros carry no grad_fn, so autograd would record no all_gather node on this
        # rank, skip its backward, and hang the others in _Reduce_Scatter. Force a leaf
        # ONLY here -- gating on `requires_grad` instead would silently detach a
        # non-empty rank whose embeddings arrived without grad.
        z_ts = torch.zeros(max_n, shared_dim, device=device, requires_grad=True)
    else:
        z_ts = z_ts.float()
        if local_n < max_n:
            pad = max_n - local_n
            z_ts = torch.cat([z_ts, z_ts.new_zeros(pad, shared_dim)], dim=0)

    g_ts = grad_all_gather(z_ts)

    # Labels are strings, so an object gather; pos_mask is not differentiable.
    # (label, family) travel as ONE list so a rank can never contribute a label
    # without its family, which would silently shift every downstream family split.
    payload = [None] * world_size
    dist.all_gather_object(payload, list(zip(list(ts_text)[:local_n], list(ts_family)[:local_n], strict=True)))

    # Slice each rank's chunk to its real row count rather than masking the whole
    # concatenation. Slicing is autograd-safe and makes label/row misalignment
    # structurally impossible -- one list drives both.
    per_rank = [list(entries)[:max_n] for entries in payload]
    keep_ts = [chunk[: len(entries)] for chunk, entries in zip(g_ts, per_rank, strict=True)]
    labels = [t for entries in per_rank for t, _ in entries]
    families = [f for entries in per_rank for _, f in entries]
    return torch.cat(keep_ts), labels, families


def _family_prototype_siglip_loss(
    z_ts: torch.Tensor,
    labels: list[str],
    families: list[str],
    prototype_bank: dict[str, tuple[tuple[str, ...], torch.Tensor]],
    enabled_families: tuple[str, ...],
    log_logit_scale: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[float], dict[str, float]]:
    """Average class-balanced SigLIP losses over fixed family vocabularies."""
    family_losses: list[torch.Tensor] = []
    positive_rates: list[float] = []
    coverage: dict[str, float] = {}
    for family in enabled_families:
        global_rows = [i for i, value in enumerate(families) if value == family]
        if not global_rows:
            continue
        entry = prototype_bank.get(family)
        if entry is None:
            raise ValueError(f"prototype family {family!r} has no fixed vocabulary")
        prototype_labels, prototypes = entry
        num_classes = len(prototype_labels)
        if num_classes < 2:
            raise ValueError(f"prototype family {family!r} needs at least two labels, got {num_classes}")

        label_to_id = {label: idx for idx, label in enumerate(prototype_labels)}
        known_rows = [i for i in global_rows if labels[i] in label_to_id]
        coverage[family] = len(known_rows) / len(global_rows)
        if not known_rows:
            continue

        select = torch.tensor(known_rows, device=z_ts.device, dtype=torch.long)
        anchors = z_ts.index_select(0, select)
        targets = torch.tensor(
            [label_to_id[labels[i]] for i in known_rows],
            device=z_ts.device,
            dtype=torch.long,
        )
        pos_mask = targets[:, None] == torch.arange(num_classes, device=z_ts.device)[None, :]

        # Each class contributes the same total anchor weight, irrespective of its
        # count. Prototypes are already unique, so every anchor also sees each class
        # exactly once on the text side.
        class_count = torch.bincount(targets, minlength=num_classes).float()
        row_weight = class_count.index_select(0, targets).reciprocal().unsqueeze(1)
        positive_rate = 1.0 / num_classes
        family_bias = anchors.new_tensor(math.log(positive_rate / (1.0 - positive_rate)))
        family_losses.append(
            siglip_sigmoid(
                anchors,
                prototypes,
                log_logit_scale,
                family_bias,
                pos_mask=pos_mask,
                pair_weight=row_weight,
            )
        )
        positive_rates.append(positive_rate)

    if not family_losses:
        return None, None, coverage
    return (
        torch.stack(family_losses).mean(),
        sum(positive_rates) / len(positive_rates),
        coverage,
    )


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    device: torch.device,
    visual_dtype: torch.dtype,
    world_size: int = 1,
    ts_eval_collector: Optional[dict[str, list]] = None,
    prototype_bank: Optional[dict[str, TextPrototypeFamily]] = None,
    gcms_bank: Optional[SmellNetGCMSBank] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute prototype SigLIP and frozen-Qwen image-preservation losses."""
    metrics: dict[str, float] = {}

    feat_img, feat_ref_img, img_rows_per_sample = _encode_branch(
        model,
        batch["img_image_pil"],
        batch["img_video"],
        device,
        visual_dtype,
    )
    feat_ts = None
    ts_signal = batch.get("ts_signal")
    if ts_signal:
        ts_formats = list(batch.get("ts_format") or [])
        crop_steps = dict(cfg.data.get("temporal_crop_steps") or {})
        if model.training and crop_steps:
            missing = sorted(set(ts_formats) - set(crop_steps))
            if missing:
                raise ValueError(f"temporal_crop_steps missing signal families {missing}")
            ts_signal = [
                temporal_crop(signal, family, int(crop_steps[family]))
                for signal, family in zip(ts_signal, ts_formats, strict=True)
            ]
        feat_ts = model.encode_ts_trainable(ts_signal, ts_formats, device=device)

    ts_text = list(batch["ts_signal_text"])
    ts_family = list(batch.get("ts_format") or [])

    feat_img, feat_ref_img = _sanitize_features(feat_img, feat_ref_img)
    feat_ts, _ = _sanitize_features(feat_ts, None)

    total = torch.zeros((), device=device, dtype=torch.float32)
    w = cfg.loss_weights

    z_ts = model.project(model.proj_visual, feat_ts) if feat_ts is not None else None
    if world_size > 1:
        c_ts, c_labels, c_family = _gather_ts_embeddings(
            z_ts,
            ts_text,
            ts_family,
            device,
            world_size,
            shared_dim=int(model.shared_dim),
        )
    else:
        c_ts, c_labels, c_family = z_ts, list(ts_text), list(ts_family)

    if c_ts is not None and c_ts.shape[0] > 0:
        if prototype_bank is None:
            raise ValueError("TS alignment requires a fixed prototype_bank")
        projected_families = set(c_family)
        if gcms_bank is not None:
            projected_families.add("smell")
        projected_bank = _project_text_prototype_bank(
            model,
            prototype_bank,
            projected_families,
        )
        enabled_families = tuple(str(value) for value in (cfg.loss.get("prototype_families") or _TS_FAMILIES))
        l_ts, _, _ = _family_prototype_siglip_loss(
            c_ts,
            c_labels,
            c_family,
            projected_bank,
            enabled_families,
            model.log_logit_scale,
        )
        if l_ts is not None:
            total = total + float(w.ts_text) * l_ts
            metrics["loss/ts_text"] = l_ts.detach().item()

        if gcms_bank is not None:
            text_smell = projected_bank.get("smell")
            if text_smell is None:
                raise ValueError("GC-MS alignment requires the SmellNet text vocabulary")
            projected_gcms = _project_gcms_bank(model, gcms_bank)

            sensor_gcms_weight = float(w.get("smell_sensor_gcms", 0.0))
            if sensor_gcms_weight > 0.0:
                l_sensor_gcms, _, _ = _family_prototype_siglip_loss(
                    c_ts,
                    c_labels,
                    c_family,
                    {"smell": projected_gcms},
                    ("smell",),
                    model.log_logit_scale,
                )
                if l_sensor_gcms is not None:
                    total = total + sensor_gcms_weight * l_sensor_gcms
                    metrics["loss/smell_sensor_gcms"] = float(l_sensor_gcms.detach())

            gcms_text_weight = float(w.get("smell_gcms_text", 0.0))
            if gcms_text_weight > 0.0:
                l_gcms_text = _paired_prototype_siglip_loss(
                    projected_gcms,
                    text_smell,
                    model.log_logit_scale,
                )
                total = total + gcms_text_weight * l_gcms_text
                metrics["loss/smell_gcms_text"] = float(l_gcms_text.detach())

            metrics.update(
                _smellnet_gcms_top1_metrics(
                    c_ts,
                    c_labels,
                    c_family,
                    text_smell,
                    projected_gcms,
                )
            )
        metrics.update(
            _ts_prediction_metrics(
                c_ts,
                c_labels,
                c_family,
                projected_bank,
            )
        )
        if ts_eval_collector is not None:
            ts_eval_collector["z"].append(c_ts.detach().float())
            ts_eval_collector["labels"].extend(c_labels)
            ts_eval_collector["families"].extend(c_family)

    if feat_img is not None and feat_ref_img is not None and feat_img.shape[0] > 0:
        l_img = distill_cosine(
            model._norm(feat_img.float()),
            model._norm(feat_ref_img.float()),
            rows_per_sample=img_rows_per_sample,
        )
        total = total + float(w.distill_img) * l_img
        metrics["loss/distill_img"] = l_img.detach().item()

    metrics["loss/total"] = total.detach().item()
    metrics["logit_scale"] = model.log_logit_scale.detach().exp().item()
    return total, metrics
