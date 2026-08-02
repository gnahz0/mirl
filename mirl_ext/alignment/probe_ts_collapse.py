# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Measure raster quality and embedding collapse for one time-series family.

    srun -p b200-devel --gres=gpu:1 -c 8 --mem=64G --time=00:30:00 \\
      python -m mirl_ext.alignment.probe_ts_collapse \\
        --family smellnet --ckpt /scratch/.../best/alignment_state.pt

"""

from __future__ import annotations

import argparse
import collections
import logging
import math
from pathlib import Path

import torch

logger = logging.getLogger("probe_ts_collapse")

# data_source values per family, as they appear in the parquet indexes.
FAMILIES: dict[str, list[str]] = {
    "smellnet": ["smellnet_base"],
    # One data_source for all four ECG corpora; provenance lives in the file path.
    "ecg": ["ecg"],
    "haptic": ["haptic_tactile"],
}


def _cos_stats(z: torch.Tensor, labels: list[str]) -> dict:
    """Compute cosine separation and effective dimension for normalized rows."""
    z = torch.nn.functional.normalize(z.float(), dim=-1, eps=1e-6)
    sims = z @ z.t()
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = torch.tensor([[a == b for b in labels] for a in labels], dtype=torch.bool, device=z.device)
    offdiag = sims[~eye]
    within = sims[same & ~eye]
    between = sims[~same]
    # Effective dimensionality: participation ratio of the covariance spectrum.
    # A collapsed set concentrates all variance in a couple of directions, so this
    # falls far below min(n, d) even when offdiag cosine looks unremarkable.
    centered = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered.float())
    ev = sv**2
    eff_dim = float((ev.sum() ** 2) / (ev**2).sum()) if float(ev.sum()) > 0 else 0.0
    return {
        "n": n,
        "n_labels": len(set(labels)),
        "offdiag_mean": float(offdiag.mean()) if offdiag.numel() else float("nan"),
        "offdiag_std": float(offdiag.std()) if offdiag.numel() > 1 else float("nan"),
        "within_mean": float(within.mean()) if within.numel() else float("nan"),
        "between_mean": float(between.mean()) if between.numel() else float("nan"),
        "margin": (float(within.mean()) - float(between.mean()))
        if within.numel() and between.numel()
        else float("nan"),
        "eff_dim": eff_dim,
    }


def _print_stats(title: str, st: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  rows={st['n']}  distinct labels={st['n_labels']}  effective dim={st['eff_dim']:.1f}")
    print(f"  offdiag cosine   mean={st['offdiag_mean']:+.4f}  std={st['offdiag_std']:.4f}")
    print(f"  within-label     mean={st['within_mean']:+.4f}")
    print(f"  between-label    mean={st['between_mean']:+.4f}")
    print(f"  MARGIN (w - b)   {st['margin']:+.4f}")
    if not math.isfinite(st["margin"]):
        print("  VERDICT: UNSCORABLE -- no repeated labels provide within-class pairs.")
    elif st["offdiag_mean"] > 0.9:
        print("  VERDICT: COLLAPSED -- distinct inputs map to near-identical vectors.")
    elif st["margin"] < 0.01:
        print("  VERDICT: NO MARGIN -- vectors differ but not along the label axis.")
    else:
        print("  VERDICT: separable -- a margin exists for the loss to exploit.")


def _raster_stats(model, signals: list, formats: list[str]) -> None:
    """Report padding, saturation, variance, grid size, and token count."""
    print("\n=== rendered raster diagnostics ===")
    pad_fracs, sat_fracs, stds, grids, tokens = [], [], [], [], []
    for sig, fmt in zip(signals, formats, strict=True):
        if fmt == "tactile":
            pixel_values, grid_thw = model._tactile_to_video_inputs(sig)
        else:
            normalization = model.ecg_normalization if fmt == "ecg" else "robust"
            pixel_values, grid_thw = model._timeseries_to_video_inputs(sig, normalization=normalization)

        # Inspect the exact flattened tensor sent to Qwen. Patchification is a pure
        # permutation/reshape, so padding, saturation, and variance are identical to
        # the source pseudo-video without duplicating renderer logic here.
        flat = pixel_values.float()
        pad_fracs.append(float((flat <= -0.999).float().mean()))
        sat_fracs.append(float((flat.abs() >= 0.99).float().mean()))
        stds.append(float(flat.std()))
        grids.append(tuple(int(v) for v in grid_thw[0].tolist()))
        tokens.append(int(grid_thw[0].prod()) // (model.vit_merge_size**2))

    def _mean(x):
        return sum(x) / len(x) if x else float("nan")

    print(f"  padding pixels (== -1):  mean {_mean(pad_fracs):.1%}  max {max(pad_fracs):.1%}")
    print(f"  saturated pixels (|v|>=.99): mean {_mean(sat_fracs):.1%}  max {max(sat_fracs):.1%}")
    print(f"  pixel std within raster: mean {_mean(stds):.4f}  min {min(stds):.4f}")
    print(f"  Qwen pre-merge grid (t,h,w; first 5): {grids[:5]}")
    print(f"  post-merger tokens per sample: mean {_mean(tokens):.1f}  min {min(tokens)}  max {max(tokens)}")
    if _mean(pad_fracs) > 0.25:
        print("  WARNING: >25% of the raster is padding -- mean-pooling dilutes real content.")
    if _mean(sat_fracs) > 0.25:
        print("  WARNING: >25% saturated -- suspect a dead channel hitting the 1e-6 scale floor.")


def _per_channel_report(signals: list, formats: list[str], limit: int = 8) -> None:
    """Report raw per-channel scale before normalization."""
    print(f"\n=== raw per-channel scale (pre-normalization, first {limit} samples) ===")
    for i, (sig, fmt) in enumerate(zip(signals[:limit], formats[:limit], strict=True)):
        if fmt == "tactile" or not isinstance(sig, torch.Tensor) or sig.ndim != 2:
            continue
        x = torch.nan_to_num(sig.float())
        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1).values / 0.6745
        std = x.std(dim=-1, unbiased=False)
        scale = 0.7 * mad + 0.3 * std
        dead = int((scale < 1e-6).sum())
        print(
            f"  [{i}] C={x.shape[0]} T={x.shape[1]}  "
            f"scale min={float(scale.min()):.3e} max={float(scale.max()):.3e}  "
            f"|median| max={float(med.abs().max()):.3e}  dead(<1e-6)={dead}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--ckpt", default=None, help="checkpoint directory or alignment_state.pt")
    ap.add_argument(
        "--config",
        default=None,
        help="config YAML (default: checkpoint sibling, then the production config)",
    )
    ap.add_argument("--samples", type=int, default=96, help="recordings to embed")
    ap.add_argument("--batch", type=int, default=8, help="VE forward batch size")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    from omegaconf import OmegaConf

    from mirl_ext.alignment.data import AlignmentDataset
    from mirl_ext.alignment.model import MultimodalAlignmentModel

    config_path = Path(args.config) if args.config else Path("mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml")
    if args.config is None and args.ckpt:
        ckpt_path = Path(args.ckpt)
        candidate = (ckpt_path if ckpt_path.is_dir() else ckpt_path.parent) / "config.yaml"
        if candidate.is_file():
            config_path = candidate
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = FAMILIES[args.family]
    files = list(cfg.data.val_files if args.split == "val" else cfg.data.train_files)

    print("=" * 78)
    print(f"ts collapse probe: family={args.family} sources={sources}")
    print(f"  ckpt = {args.ckpt or '<PRISTINE Qwen3.5 tower>'}")
    print(f"  config = {config_path}")
    print("=" * 78)

    ds = AlignmentDataset(
        data_files=files,
        text_for_label=cfg.data.get("text_for_label", "ground_truth"),
        tactile_label_mode=str(cfg.data.get("tactile_label_mode", "ground_truth")),
        max_samples=-1,
        seed=cfg.train.get("seed", 42),
        enable_videos=False,  # ts branch only; skip video decode entirely
        data_source_filter=sources,
        exclude_data_sources=list(cfg.data.get("exclude_data_sources") or []) or None,
        tactile_max_frames=cfg.data.get("tactile_max_frames"),
        include_all_ts=True,
    )
    print(f"\ndataset: {len(ds)} rows for {sources}")

    signals, formats, texts = [], [], []
    for i in range(len(ds)):
        if len(signals) >= args.samples:
            break
        item = ds[i]
        if item["branch"] != "ts" or item["media"] is None:
            continue
        signals.append(item["media"])
        formats.append(item.get("ts_format", "smell"))
        texts.append(item["text"])
    print(f"loaded {len(signals)} signals, {len(set(texts))} distinct labels")
    if len(signals) < 4:
        raise SystemExit("too few signals loaded to say anything")

    model = MultimodalAlignmentModel(
        qwen35_path=str(cfg.model.qwen35_path),
        siglip2_text_path=str(cfg.model.siglip2_text_path),
        shared_dim=int(cfg.projection.shared_dim),
        proj_hidden_dim=(int(cfg.projection.hidden_dim) if cfg.projection.get("hidden_dim") is not None else None),
        proj_dropout=0.0,
        visual_dtype=torch.bfloat16,
        attn_impl=str(cfg.model.get("attn_impl", "sdpa")),
        ecg_normalization=str(cfg.model.get("ecg_normalization", "robust")),
        tactile_delta_channels=bool(cfg.model.get("tactile_delta_channels", False)),
    )
    if args.ckpt:
        from mirl_ext.alignment.runtime import load_checkpoint

        load_checkpoint(model, Path(args.ckpt))
    model.to(device).eval()

    # Raster/channel diagnostics are CPU-only and independent of the checkpoint.
    _per_channel_report(signals, formats)
    _raster_stats(model, signals, formats)

    feats = []
    with torch.no_grad():
        for start in range(0, len(signals), args.batch):
            chunk = signals[start : start + args.batch]
            fmt_chunk = formats[start : start + args.batch]
            f = model.encode_ts_trainable(chunk, fmt_chunk, device=device)
            feats.append(f.float().cpu())
    feat = torch.cat(feats, dim=0)
    print(f"\npooled VE features: {tuple(feat.shape)}")

    with torch.no_grad():
        z_vis = model.project(model.proj_visual, feat.to(device)).float().cpu()
        t_raw = model.encode_text(texts, device=device).float()
        z_txt = model.project(model.proj_text, t_raw).float().cpu()

    print("\n" + "=" * 78)
    print("HYPOTHESIS 1 -- VE collapse (the ceiling on what any head can do)")
    _print_stats("raw pooled Qwen VE features (4096-d, pre-projection)", _cos_stats(feat, texts))
    print("\n" + "=" * 78)
    print("HYPOTHESIS 2 -- text-side collapse (do the targets have margin?)")
    _print_stats("raw SigLIP2 label embeddings", _cos_stats(t_raw.cpu(), texts))
    print("\n" + "=" * 78)
    print("HYPOTHESIS 4 -- the projected space the loss actually optimizes")
    _print_stats("projected visual (512-d)", _cos_stats(z_vis, texts))
    _print_stats("projected text (512-d)", _cos_stats(z_txt, texts))

    # Cross-modal: this is exactly what accuracy/F1 argmax over prototypes reads.
    zv = torch.nn.functional.normalize(z_vis, dim=-1, eps=1e-6)
    zt = torch.nn.functional.normalize(z_txt, dim=-1, eps=1e-6)
    sims = zv @ zt.t()
    same = torch.tensor([[a == b for b in texts] for a in texts], dtype=torch.bool)
    pos = sims[same].mean()
    neg = sims[~same].mean()
    label_to_idx: dict[str, int] = {}
    proto_rows: list[int] = []
    true_ids: list[int] = []
    for i, t in enumerate(texts):
        if t not in label_to_idx:
            label_to_idx[t] = len(label_to_idx)
            proto_rows.append(i)
        true_ids.append(label_to_idx[t])
    pred = (zv @ zt[proto_rows].t()).argmax(dim=1)
    acc = float((pred == torch.tensor(true_ids)).float().mean())
    # Uniform 1/K is the WRONG yardstick for a skewed label set. ECG's 7 classes are
    # 44% "Normal", so a constant majority predictor already scores 0.44 and uniform
    # chance (0.14) flatters the result by ~3x. Report both and compare against the
    # larger one.
    counts = collections.Counter(true_ids)
    majority = counts.most_common(1)[0][1] / len(true_ids)
    uniform = 1.0 / len(proto_rows)
    baseline = max(majority, uniform)
    print("\n" + "=" * 78)
    print("CROSS-MODAL (what accuracy/F1 measures)")
    print(f"  pos_mean={float(pos):+.4f}  neg_mean={float(neg):+.4f}  gap={float(pos - neg):+.4f}")
    print(f"  top-1 acc over {len(proto_rows)} prototypes = {acc:.4f}")
    print(f"    uniform-chance baseline      = {uniform:.4f}")
    print(f"    MAJORITY-class baseline      = {majority:.4f}  <- the honest one")
    print(f"    margin over stronger baseline= {acc - baseline:+.4f}")
    print(f"  predicted distinct classes: {len(set(pred.tolist()))} of {len(proto_rows)}")
    if acc - baseline < 0.05:
        print("  VERDICT: at or near the trivial baseline -- no usable alignment.")
    if len(set(pred.tolist())) <= 2:
        print("  WARNING: predictions concentrate on <=2 prototypes -- classic collapse signature.")


if __name__ == "__main__":
    main()
