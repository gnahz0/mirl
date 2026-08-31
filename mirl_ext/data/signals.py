"""Raw sensor IO and pseudo-video rendering, shared across stages.

Moved verbatim from the alignment package so the same loaders and the same
signal -> Qwen-video-tile render are available to Stage-1 alignment today and
to a native-signal SFT/RL student path later. Rendering: [C, T] series become
one merger-cell-wide tile column per frame; [T, 16, 16] tactile maps become
one nearest-resized frame per timestep. Series are z-scored per channel over
time (already-z-scored ECG passes through unchanged); tactile is z-scored
over the whole recording. Both clamp at 4 sigma and scale to [-1, 1].
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

SIGNAL_FORMATS = {"": "smellnet", "ts_pt": "ecg", "tactile_pt": "tactile"}


def signal_family(sig_entry: dict) -> str:
    return SIGNAL_FORMATS[sig_entry["format"]]


def load_signal_csv(path: str) -> torch.Tensor:
    """SmellNet CSV -> [C, T] float tensor, time columns dropped."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    keep_idx = [i for i, name in enumerate(header) if "time" not in name.casefold()]
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=keep_idx,
        dtype=np.float32,
    )
    return torch.from_numpy(np.ascontiguousarray(data.T))


def load_signal(sig_entry: dict) -> tuple[torch.Tensor, str]:
    """One signals[] entry -> (tensor, family). ecg: [8, 2500]; tactile:
    [T, 16, 16] selected by key; smellnet: [C, T] from CSV."""
    family = signal_family(sig_entry)
    path = sig_entry["signal"]
    if family == "ecg":
        signal = torch.load(path, map_location="cpu", weights_only=False)
        return signal.float().contiguous(), family
    if family == "tactile":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        signal = torch.as_tensor(payload["tactile"][sig_entry["key"]])
        return signal.float().contiguous(), family
    return load_signal_csv(path), family


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-score each row over the last dim, clamped at 4 sigma, scaled to [-1, 1].

    Constant rows map to zero (zero numerator against the clamped std)."""
    if not torch.isfinite(x).all():
        raise ValueError("signal contains non-finite values")

    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return ((x - mean) / std.clamp_min(1e-6)).clamp(-4.0, 4.0) / 4.0


def timeseries_frames(signal: torch.Tensor, cell: int) -> torch.Tensor:
    """Pack consecutive cell-width [C, T] blocks as video frames. ``cell`` is
    the ViT patch size times the spatial merge size (one merged token)."""
    value = normalize(signal)

    channels, steps = value.shape
    frame_count = math.ceil(steps / cell)
    padded = F.pad(value, (0, frame_count * cell - steps), value=-1.0)
    tiles = padded.reshape(channels, frame_count, cell).permute(1, 0, 2)
    tiles = tiles.repeat_interleave(cell, dim=1)
    return tiles.unsqueeze(1).expand(-1, 3, -1, -1)


def tactile_frames(tactile: torch.Tensor, side: int) -> torch.Tensor:
    """Z-score the whole recording and resize each [16, 16] map to one
    ``side`` x ``side`` merger cell.

    One global scale keeps the spatial pressure pattern (OpenTouch's recipe);
    per-taxel scaling would render untouched taxels' noise at full amplitude."""
    value = normalize(tactile.float().reshape(1, -1)).reshape_as(tactile)
    value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest")
    return value.expand(-1, 3, -1, -1)


def family_frames(signal: torch.Tensor, family: str, cell: int) -> torch.Tensor:
    """The family -> renderer dispatch shared by the ts-native builder and
    Stage-1 (``no duplicated math`` invariant, rl/ts_native_DESIGN.md)."""
    return tactile_frames(signal, cell) if family == "tactile" else timeseries_frames(signal, cell)
