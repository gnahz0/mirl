# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Stage-1 alignment objective, text-label banks, and evaluation."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributed.nn.functional import all_gather as grad_all_gather
from torch.utils.data import DataLoader

from .losses import distill_cosine, siglip_sigmoid
from .metrics import (
    _TS_FAMILIES,
    _allreduce_counts,
    _allreduce_metrics,
    _ts_prediction_metrics,
    _validation_metric_groups,
    add_ts_family_counts,
    new_counts,
)
from .model import MultimodalAlignmentModel

logger = logging.getLogger("alignment.trainer")

LabelBank = dict[str, tuple[tuple[str, ...], torch.Tensor]]
TSEval = tuple[torch.Tensor | None, list[str], list[str]]


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
    max_image_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int]]:
    """Encode image/video post-merger tokens with student and frozen Qwen towers."""
    if not images_pil and not videos:
        return None, None, []

    if images_pil:
        image_kwargs = {"return_tensors": "pt"}
        token_pixels = (model.vit_patch_size * model.vit_merge_size) ** 2
        max_pixels = max_image_tokens * token_pixels
        min_pixels = min(int(model.qwen_processor.image_processor.size["shortest_edge"]), max_pixels)
        image_kwargs["size"] = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        processed = model.qwen_processor.image_processor.preprocess(
            images_pil,
            **image_kwargs,
        )
        pixel_key, grid_key, kind = "pixel_values", "image_grid_thw", "image"
    else:
        tensors, metadata = zip(*videos, strict=True)
        processed = model.qwen_processor.video_processor.preprocess(
            list(tensors),
            video_metadata=list(metadata),
            do_sample_frames=False,
            return_tensors="pt",
        )
        pixel_key, grid_key, kind = "pixel_values_videos", "video_grid_thw", "video"

    pixels = processed[pixel_key].to(device=device)
    grid = processed[grid_key].to(device=device)
    _log_visual_batch_once(model, kind, grid)
    rows_per_sample = grid.prod(dim=1).tolist()
    return (
        model.encode_visual(pixels, grid, pool=False),
        model.encode_visual(pixels, grid, frozen=True, pool=False),
        rows_per_sample,
    )


@torch.no_grad()
def _build_text_label_bank(
    model: MultimodalAlignmentModel,
    vocabularies: dict[str, tuple[str, ...]],
    device: torch.device,
    batch_size: int = 256,
) -> LabelBank:
    """Encode each split's complete family label vocabulary once with SigLIP2."""
    bank: LabelBank = {}
    for family in _TS_FAMILIES:
        labels = tuple(vocabularies.get(family, ()))
        if not labels:
            continue
        chunks = [
            model.encode_text(list(labels[start : start + batch_size]), device=device).float()
            for start in range(0, len(labels), batch_size)
        ]
        bank[family] = (labels, torch.cat(chunks, dim=0).detach())
        logger.info("text label bank: family=%s labels=%d", family, len(labels))
    return bank


@torch.no_grad()
def _score_ts(
    embeddings: list[torch.Tensor],
    labels: list[str],
    families: list[str],
    label_bank: LabelBank,
    per_class_reports: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, float]:
    """Score one sensor sample set against the frozen text-label bank."""
    if not embeddings:
        return {}
    return _ts_prediction_metrics(
        torch.cat(embeddings),
        labels,
        families,
        label_bank,
        per_class_reports=per_class_reports,
    )


@torch.no_grad()
def _run_validation(
    model: MultimodalAlignmentModel,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    label_bank: LabelBank,
    world_size: int = 1,
    metadata_group: dist.ProcessGroup | None = None,
) -> tuple[
    dict[str, float],
    dict[str, list[dict[str, object]]],
]:
    """Evaluate losses and prediction metrics over validation batches."""
    was_training = model.training
    model.eval()

    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    # Collect embeddings once, then score the validation set globally.
    ts_embeddings: list[torch.Tensor] = []
    ts_labels: list[str] = []
    ts_families: list[str] = []
    bucket_totals = new_counts()
    for batch in val_loader:
        n_ts = len(batch["ts_signal"])
        bucket_totals["n/img_image"] += len(batch["img_image_pil"])
        bucket_totals["n/img_video"] += len(batch["img_video"])
        bucket_totals["n/ts_signal"] += n_ts
        add_ts_family_counts(bucket_totals, batch)
        with autocast_ctx:
            _, metrics, batch_eval = _compute_losses(
                model,
                batch,
                cfg,
                device,
                world_size=world_size,
                label_bank=label_bank,
                metadata_group=metadata_group,
            )
        embeddings, labels, families = batch_eval
        if embeddings is not None:
            ts_embeddings.append(embeddings)
            ts_labels.extend(labels)
            ts_families.extend(families)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    averaged = _allreduce_metrics(
        {key: sums[key] / counts[key] for key in sums},
        device,
        world_size,
    )
    bucket_totals = _allreduce_counts(bucket_totals, device, world_size)
    per_class_reports: dict[str, list[dict[str, object]]] = {}
    prediction_metrics = _score_ts(
        ts_embeddings,
        ts_labels,
        ts_families,
        label_bank,
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
    z_ts: torch.Tensor | None,
    labels: list[str],
    families: list[str],
    device: torch.device,
    world_size: int,
    embedding_dim: int,
    metadata_group: dist.ProcessGroup | None = None,
):
    """Differentiably gather uneven TS rows and metadata from every rank."""
    if world_size <= 1:
        return z_ts, labels, families

    empty = z_ts is None
    local_n = 0 if empty else int(z_ts.shape[0])

    # Agree on the padded width. All ranks, unconditionally.
    n_t = torch.tensor([local_n], device=device, dtype=torch.long)
    dist.all_reduce(n_t, op=dist.ReduceOp.MAX)
    max_n = int(n_t.item())
    if max_n == 0:
        return None, [], []

    # Use fp32 everywhere so empty and non-empty ranks agree on collective dtype.
    if empty:
        # Zeros carry no grad_fn, so autograd would record no all_gather node on this
        # rank, skip its backward, and hang the others in _Reduce_Scatter. Force a leaf
        # ONLY here -- gating on `requires_grad` instead would silently detach a
        # non-empty rank whose embeddings arrived without grad.
        z_ts = torch.zeros(max_n, embedding_dim, device=device, requires_grad=True)
    else:
        z_ts = z_ts.float()
        if local_n < max_n:
            pad = max_n - local_n
            z_ts = torch.cat([z_ts, z_ts.new_zeros(pad, embedding_dim)], dim=0)

    g_ts = grad_all_gather(z_ts)

    # Labels are strings, so an object gather; pos_mask is not differentiable.
    # (label, family) travel as ONE list so a rank can never contribute a label
    # without its family, which would silently shift every downstream family split.
    payload = [None] * world_size
    dist.all_gather_object(
        payload,
        list(zip(labels[:local_n], families[:local_n], strict=True)),
        group=metadata_group,
    )

    # Slice each rank's chunk to its real row count rather than masking the whole
    # concatenation. Slicing is autograd-safe and makes label/row misalignment
    # structurally impossible -- one list drives both.
    per_rank = [list(entries)[:max_n] for entries in payload]
    keep_ts = [chunk[: len(entries)] for chunk, entries in zip(g_ts, per_rank, strict=True)]
    labels = [t for entries in per_rank for t, _ in entries]
    families = [f for entries in per_rank for _, f in entries]
    return torch.cat(keep_ts), labels, families


def _family_label_siglip_loss(
    z_ts: torch.Tensor,
    labels: list[str],
    families: list[str],
    label_bank: dict[str, tuple[tuple[str, ...], torch.Tensor]],
    log_logit_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute label-balanced SigLIP against each family's complete text bank."""
    family_losses: dict[str, torch.Tensor] = {}
    for family, (candidate_labels, text_embeddings) in label_bank.items():
        global_rows = [i for i, value in enumerate(families) if value == family]
        if not global_rows:
            continue
        num_labels = len(candidate_labels)
        label_to_id = {label: idx for idx, label in enumerate(candidate_labels)}
        select = torch.tensor(global_rows, device=z_ts.device, dtype=torch.long)
        anchors = z_ts.index_select(0, select)
        targets = torch.tensor(
            [label_to_id[labels[i]] for i in global_rows],
            device=z_ts.device,
            dtype=torch.long,
        )
        pos_mask = targets[:, None] == torch.arange(num_labels, device=z_ts.device)[None, :]

        # Repeated labels contribute the same total anchor weight as unique labels.
        class_count = torch.bincount(targets, minlength=num_labels).float()
        row_weight = class_count.index_select(0, targets).reciprocal().unsqueeze(1)
        positive_rate = 1.0 / num_labels
        family_bias = anchors.new_tensor(math.log(positive_rate / (1.0 - positive_rate)))
        family_losses[family] = siglip_sigmoid(
            anchors,
            text_embeddings,
            log_logit_scale,
            family_bias,
            pos_mask=pos_mask,
            pair_weight=row_weight,
        )

    return family_losses


def _compute_losses(
    model: MultimodalAlignmentModel,
    batch: dict,
    cfg: DictConfig,
    device: torch.device,
    *,
    label_bank: LabelBank,
    world_size: int = 1,
    metadata_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, dict, TSEval]:
    """Compute family label-bank SigLIP and frozen-Qwen preservation losses."""
    metrics: dict[str, float] = {}

    feat_img, feat_ref_img, img_rows_per_sample = _encode_branch(
        model,
        batch["img_image_pil"],
        batch["img_video"],
        device,
        max_image_tokens=int(cfg.data.max_image_tokens),
    )
    feat_ts = None
    ts_signal = batch["ts_signal"]
    ts_family = list(batch["ts_format"])
    ts_labels = list(batch["ts_signal_text"])
    if ts_signal:
        feat_ts = model.encode_ts_trainable(ts_signal, ts_family, device=device)

    total = torch.zeros((), device=device, dtype=torch.float32)

    z_ts = F.normalize(feat_ts.float(), dim=-1, eps=1e-6) if feat_ts is not None else None
    c_ts, c_labels, c_family = _gather_ts_embeddings(
        z_ts,
        ts_labels,
        ts_family,
        device,
        world_size,
        embedding_dim=int(model.trainable_visual.config.hidden_size),
        metadata_group=metadata_group,
    )

    if c_ts is not None:
        family_ts_losses = _family_label_siglip_loss(
            c_ts,
            c_labels,
            c_family,
            label_bank,
            model.log_logit_scale,
        )
        l_ts = torch.stack(tuple(family_ts_losses.values())).mean()
        total = total + float(cfg.loss.siglip_weight) * l_ts
        metrics["loss/siglip"] = l_ts.detach().item()
        metrics.update({
            f"loss/ts_{family}": family_loss.detach().item()
            for family, family_loss in family_ts_losses.items()
        })

    if feat_img is not None and feat_ref_img is not None and feat_img.shape[0] > 0:
        l_img = distill_cosine(
            feat_img,
            feat_ref_img,
            rows_per_sample=img_rows_per_sample,
        )
        total = total + float(cfg.loss.distill_weight) * l_img
        metrics["loss/distill"] = l_img.detach().item()

    metrics["loss/total"] = total.detach().item()
    metrics["logit_scale"] = model.log_logit_scale.detach().exp().item()
    ts_eval = (c_ts.detach().float(), c_labels, c_family) if c_ts is not None else (None, [], [])
    return total, metrics, ts_eval
