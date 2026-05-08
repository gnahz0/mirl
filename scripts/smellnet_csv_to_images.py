"""
Convert SmellNet CSV time series data into image representations for vision encoder training.

Supports multiple representation modes:
  - overlay:   All channels on one plot, per-channel normalized to [0,1], with legend
  - subplot:   One subplot per channel, each labeled and independently scaled
  - heatmap:   Rows = channels, columns = time steps, pixel intensity = normalized value
               Compact raster representation — no matplotlib overhead, very fast
  - multichannel: Save as a raw N-channel numpy array (.npy) for direct tensor loading
                  (no information loss, but requires custom dataloader)

Usage:
    python scripts/smellnet_csv_to_images.py \\
        --data_dir /path/to/SmellNet \\
        --output_dir /path/to/SmellNet_images \\
        --mode overlay --img_size 224 --num_workers 16

    # Generate all modes for comparison on a single file:
    python scripts/smellnet_csv_to_images.py \\
        --data_dir /path/to/SmellNet --output_dir /tmp/compare --mode all \\
        --single /path/to/SmellNet/base_data/training/almond/almond_1.csv
"""

import argparse
import os
from multiprocessing import Pool
from functools import partial

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SENSOR_COLS_BASE = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
SENSOR_COLS_MIXTURE = ["NO2", "C2H5CH", "VOC", "CO"]
CHANNEL_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max normalize each column independently to [0, 1]."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo, hi = out[c].min(), out[c].max()
        if hi - lo < 1e-9:
            out[c] = 0.5
        else:
            out[c] = (out[c] - lo) / (hi - lo)
    return out


# ── Mode: overlay ──────────────────────────────────────────────────────────

def plot_overlay(df: pd.DataFrame, sensor_cols: list[str], out_path: str,
                 img_size: int = 224, dpi: int = 100):
    """All channels overlaid on one axes, each normalized to [0,1], with legend."""
    df_n = _normalize_columns(df, sensor_cols)
    figsize = img_size / dpi
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    x = np.arange(len(df_n))
    present = [c for c in sensor_cols if c in df_n.columns]
    for i, col in enumerate(present):
        ax.plot(x, df_n[col].values, color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                linewidth=0.8, alpha=0.9, label=col)

    ax.legend(fontsize=4, loc="upper right", framealpha=0.7, handlelength=1, labelspacing=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ── Mode: subplot ──────────────────────────────────────────────────────────

def plot_subplot(df: pd.DataFrame, sensor_cols: list[str], out_path: str,
                 img_size: int = 224, dpi: int = 100):
    """One subplot per channel, independently scaled, labeled."""
    present = [c for c in sensor_cols if c in df.columns]
    n = len(present)
    fig_w = img_size / dpi
    fig_h = fig_w * 1.2
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), dpi=dpi, sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(df))
    for i, col in enumerate(present):
        ax = axes[i]
        ax.plot(x, df[col].values, color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                linewidth=1.2)
        ax.set_ylabel(col, fontsize=10, fontweight="bold", rotation=0,
                      labelpad=45, va="center")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xticks([])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Time", fontsize=9)
    fig.suptitle("Gas Sensor Readings", fontsize=11, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.05, hspace=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ── Mode: heatmap ──────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, sensor_cols: list[str], out_path: str,
                 img_size: int = 224):
    """Raster heatmap: rows=channels, cols=time, intensity=normalized value.
    Resized to img_size x img_size. Channel names rendered as left-side labels."""
    present = [c for c in sensor_cols if c in df.columns]
    df_n = _normalize_columns(df, present)

    matrix = np.stack([df_n[c].values for c in present], axis=0)  # (n_channels, time_steps)
    matrix = (matrix * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(matrix, mode="L")
    img = img.resize((img_size, img_size), Image.NEAREST)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


# ── Mode: heatmap_labeled ──────────────────────────────────────────────────

def plot_heatmap_labeled(df: pd.DataFrame, sensor_cols: list[str], out_path: str,
                         img_size: int = 224, dpi: int = 100):
    """Heatmap via matplotlib with channel labels on the y-axis and a colorbar."""
    present = [c for c in sensor_cols if c in df.columns]
    df_n = _normalize_columns(df, present)
    matrix = np.stack([df_n[c].values for c in present], axis=0)

    figsize = img_size / dpi
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=5)
    ax.set_xticks([])
    ax.set_xlabel("time", fontsize=5)
    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── Mode: multichannel ─────────────────────────────────────────────────────

def save_multichannel(df: pd.DataFrame, sensor_cols: list[str], out_path: str,
                      img_size: int = 224):
    """Save as N-channel .npy image tensor (C, H, W) with bilinear resize.
    Each channel is one sensor, normalized to [0,1], tiled to a 2D image."""
    present = [c for c in sensor_cols if c in df.columns]
    df_n = _normalize_columns(df, present)

    matrix = np.stack([df_n[c].values for c in present], axis=0)  # (C, T)
    # tile each 1D signal into a 2D band, then resize
    bands = []
    band_h = max(1, img_size // len(present))
    for i in range(len(present)):
        band = np.tile(matrix[i], (band_h, 1))  # (band_h, T)
        img = Image.fromarray((band * 255).astype(np.uint8), mode="L")
        img = img.resize((img_size, band_h), Image.BILINEAR)
        bands.append(np.array(img, dtype=np.float32) / 255.0)

    tensor = np.stack(bands, axis=0)  # (C, band_h, W)
    out_path = os.path.splitext(out_path)[0] + ".npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, tensor)


# ── Dispatcher ─────────────────────────────────────────────────────────────

MODE_FUNCS = {
    "overlay": plot_overlay,
    "subplot": plot_subplot,
    "heatmap": plot_heatmap,
    "heatmap_labeled": plot_heatmap_labeled,
    "multichannel": save_multichannel,
}


def detect_format(csv_path: str) -> tuple[list[str], bool]:
    df_head = pd.read_csv(csv_path, nrows=2)
    cols = [c.strip() for c in df_head.columns]
    if "timestamp_ms" in cols:
        return SENSOR_COLS_MIXTURE, True
    return SENSOR_COLS_BASE, False


def process_single_csv(csv_path: str, data_dir: str, output_dir: str,
                       img_size: int, dpi: int, mode: str):
    try:
        sensor_cols, has_timestamp = detect_format(csv_path)
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if has_timestamp:
            df = df.drop(columns=["timestamp_ms"], errors="ignore")

        rel_path = os.path.relpath(csv_path, data_dir)
        ext = ".npy" if mode == "multichannel" else ".png"
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ext)

        fn = MODE_FUNCS[mode]
        kwargs = {"img_size": img_size}
        if mode in ("overlay", "subplot", "heatmap_labeled"):
            kwargs["dpi"] = dpi
        fn(df, sensor_cols, out_path, **kwargs)
        return csv_path, True, None
    except Exception as e:
        return csv_path, False, str(e)


def collect_csvs(data_dir: str) -> list[str]:
    csvs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    csvs.sort()
    return csvs


def main():
    parser = argparse.ArgumentParser(description="Convert SmellNet CSVs to image representations")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mode", type=str, default="overlay",
                        choices=list(MODE_FUNCS.keys()) + ["all"],
                        help="Image representation mode")
    parser.add_argument("--subset", type=str, default=None,
                        choices=["base_data", "mixture_data"])
    parser.add_argument("--single", type=str, default=None,
                        help="Process a single CSV (for quick comparison)")
    args = parser.parse_args()

    modes = list(MODE_FUNCS.keys()) if args.mode == "all" else [args.mode]

    if args.single:
        csv_files = [args.single]
    else:
        search_dir = args.data_dir
        if args.subset:
            search_dir = os.path.join(args.data_dir, args.subset)
        csv_files = collect_csvs(search_dir)

    print(f"Found {len(csv_files)} CSV files, modes: {modes}")

    for mode in modes:
        out_dir = os.path.join(args.output_dir, mode) if len(modes) > 1 else args.output_dir
        print(f"\n=== Mode: {mode} -> {out_dir} ===")

        worker_fn = partial(process_single_csv,
                            data_dir=args.data_dir, output_dir=out_dir,
                            img_size=args.img_size, dpi=args.dpi, mode=mode)

        success, fail = 0, 0
        if len(csv_files) == 1:
            for csv_path in csv_files:
                path, ok, err = worker_fn(csv_path)
                success += ok
                fail += (not ok)
                if err:
                    print(f"  FAILED: {path} -> {err}")
        else:
            with Pool(args.num_workers) as pool:
                for i, (path, ok, err) in enumerate(pool.imap_unordered(worker_fn, csv_files)):
                    success += ok
                    fail += (not ok)
                    if err:
                        print(f"  FAILED: {path} -> {err}")
                    if (i + 1) % 200 == 0 or (i + 1) == len(csv_files):
                        print(f"  Progress: {i+1}/{len(csv_files)}  (ok={success}, fail={fail})")

        print(f"  Done: {success} saved, {fail} failed")


if __name__ == "__main__":
    main()
