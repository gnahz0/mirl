# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Render a time-series (channels x time) array into a PIL.Image suitable for the Qwen3-VL VE.

Reuses the rendering style from ``scripts/smellnet_csv_to_images.py`` but exposed as a
runtime callable that returns ``PIL.Image`` instead of saving to disk.

In Stage 1 the smellnet samples in the existing JSONL already point at pre-rendered PNGs
under ``/home/alecz/scratch/alecz/SmellNet_subplot/``; this module is used (a) for any new
dataset that ships *raw* time-series arrays, and (b) when ``cfg.ts_render_mode != "use_existing"``
and we want to re-render on the fly with a different mode for ablation.
"""

from __future__ import annotations

import io
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CHANNEL_COLORS: tuple[str, ...] = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
)


def _normalize(ts: np.ndarray) -> np.ndarray:
    """Per-channel min-max to [0, 1]. ``ts`` is (C, T)."""
    lo = ts.min(axis=1, keepdims=True)
    hi = ts.max(axis=1, keepdims=True)
    denom = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    out = (ts - lo) / denom
    return np.where(hi - lo < 1e-9, 0.5, out)


def _as_cxt(ts: np.ndarray) -> np.ndarray:
    """Coerce input to (C, T)."""
    arr = np.asarray(ts, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"time series must be 2D, got shape {arr.shape}")
    # Heuristic: usually T >> C. If first dim is larger, transpose.
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _fig_to_pil(fig, img_size: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)
    return img


def render_subplot(ts: np.ndarray, channel_names: Optional[Sequence[str]] = None,
                   img_size: int = 224, dpi: int = 100) -> Image.Image:
    arr = _as_cxt(ts)
    c, _ = arr.shape
    figsize = img_size / dpi
    fig, axes = plt.subplots(c, 1, figsize=(figsize, figsize * 1.2), dpi=dpi, sharex=True)
    if c == 1:
        axes = [axes]
    x = np.arange(arr.shape[1])
    names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
    for i in range(c):
        ax = axes[i]
        ax.plot(x, arr[i], color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)], linewidth=1.2)
        ax.set_ylabel(names[i], fontsize=10, fontweight="bold", rotation=0,
                      labelpad=45, va="center")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xticks([])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("Time", fontsize=9)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.05, hspace=0.25)
    return _fig_to_pil(fig, img_size)


def render_overlay(ts: np.ndarray, channel_names: Optional[Sequence[str]] = None,
                   img_size: int = 224, dpi: int = 100) -> Image.Image:
    arr = _normalize(_as_cxt(ts))
    c, _ = arr.shape
    figsize = img_size / dpi
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    x = np.arange(arr.shape[1])
    names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
    for i in range(c):
        ax.plot(x, arr[i], color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                linewidth=0.9, alpha=0.9, label=names[i])
    ax.legend(fontsize=4, loc="upper right", framealpha=0.7,
              handlelength=1, labelspacing=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return _fig_to_pil(fig, img_size)


def render_heatmap(ts: np.ndarray, channel_names: Optional[Sequence[str]] = None,
                   img_size: int = 224, dpi: int = 100) -> Image.Image:
    arr = _normalize(_as_cxt(ts))
    c, _ = arr.shape
    figsize = img_size / dpi
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.imshow(arr, aspect="auto", cmap="viridis", interpolation="nearest")
    names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
    ax.set_yticks(range(c))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xticks([])
    ax.set_xlabel("time", fontsize=5)
    fig.tight_layout(pad=0.3)
    return _fig_to_pil(fig, img_size)


def render_spectrogram(ts: np.ndarray, channel_names: Optional[Sequence[str]] = None,
                       img_size: int = 224, dpi: int = 100,
                       n_fft: int = 64, hop: Optional[int] = None) -> Image.Image:
    """Tiny per-channel STFT magnitude, stacked vertically. Pure-numpy, no scipy dep."""
    arr = _as_cxt(ts)
    c, t = arr.shape
    if hop is None:
        hop = max(1, n_fft // 4)
    window = np.hanning(n_fft).astype(np.float32)

    spec_rows = []
    for i in range(c):
        x = arr[i]
        if t < n_fft:
            x = np.pad(x, (0, n_fft - t), mode="constant")
            t_eff = n_fft
        else:
            t_eff = t
        n_frames = 1 + (t_eff - n_fft) // hop
        if n_frames < 1:
            n_frames = 1
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_frames]
        frames = frames * window
        mag = np.abs(np.fft.rfft(frames, axis=-1)).T  # (n_freqs, n_frames)
        mag = np.log1p(mag)
        m = mag.max() if mag.max() > 1e-9 else 1.0
        spec_rows.append(mag / m)

    figsize = img_size / dpi
    fig, axes = plt.subplots(c, 1, figsize=(figsize, figsize * 1.2), dpi=dpi, sharex=True)
    if c == 1:
        axes = [axes]
    names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
    for i in range(c):
        axes[i].imshow(spec_rows[i], aspect="auto", origin="lower", cmap="magma")
        axes[i].set_ylabel(names[i], fontsize=6, rotation=0, labelpad=20, va="center")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[-1].set_xlabel("time", fontsize=6)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.05, hspace=0.15)
    return _fig_to_pil(fig, img_size)


_RENDERERS = {
    "subplot": render_subplot,
    "overlay": render_overlay,
    "heatmap": render_heatmap,
    "spectrogram": render_spectrogram,
}


def render_time_series_to_image(
    ts: np.ndarray,
    mode: str = "subplot",
    channel_names: Optional[Sequence[str]] = None,
    img_size: int = 224,
    dpi: int = 100,
) -> Image.Image:
    """Top-level entry. ``ts`` is a 2D ndarray of shape (channels, time) or (time, channels).

    Modes: ``subplot`` (default, one row per channel),
    ``overlay`` (per-channel normalized, single axes),
    ``heatmap`` (rows = channels, cols = time, cmap=viridis),
    ``spectrogram`` (per-channel STFT magnitude, stacked).
    """
    if mode not in _RENDERERS:
        raise ValueError(f"unknown ts render mode {mode!r}; choose from {list(_RENDERERS)}")
    return _RENDERERS[mode](ts, channel_names=channel_names, img_size=img_size, dpi=dpi)
