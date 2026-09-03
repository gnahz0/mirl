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

import torch
import torch.nn.functional as F

SIGNAL_FORMATS = {"ts_pt": "ecg", "tactile_pt": "tactile"}
SIGNAL_FAMILIES = frozenset(SIGNAL_FORMATS.values())


def signal_family(sig_entry: dict) -> str:
    """Resolve a serialized format without silently routing unknown inputs."""
    signal_format = sig_entry.get("format")
    try:
        return SIGNAL_FORMATS[signal_format]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"unsupported signal format {signal_format!r}; expected one of {tuple(SIGNAL_FORMATS)}"
        ) from exc


def load_signal(sig_entry: dict) -> tuple[torch.Tensor, str]:
    """One signals[] entry -> (tensor, family). ecg: [8, 2500]; tactile:
    [T, 16, 16] selected by key."""
    family = signal_family(sig_entry)
    path = sig_entry["signal"]
    if family == "ecg":
        signal = torch.load(path, map_location="cpu", weights_only=False)
        signal = torch.as_tensor(signal)
        if signal.ndim != 2 or any(size <= 0 for size in signal.shape):
            raise ValueError(f"ecg signal must have shape [channels, time], got {tuple(signal.shape)}")
        return signal.float().contiguous(), family
    if family == "tactile":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        try:
            signal = torch.as_tensor(payload["tactile"][sig_entry["key"]])
        except (KeyError, TypeError) as exc:
            raise ValueError("tactile_pt needs a payload tactile map and signals[].key") from exc
        if signal.ndim != 3 or any(size <= 0 for size in signal.shape):
            raise ValueError(f"tactile signal must have shape [time, height, width], got {tuple(signal.shape)}")
        return signal.float().contiguous(), family
    raise ValueError(f"unsupported signal family {family!r}")


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
    if signal.ndim != 2 or any(size <= 0 for size in signal.shape):
        raise ValueError(f"time series must have shape [channels, time], got {tuple(signal.shape)}")
    if cell <= 0:
        raise ValueError(f"cell must be positive, got {cell}")
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
    if tactile.ndim != 3 or any(size <= 0 for size in tactile.shape):
        raise ValueError(f"tactile signal must have shape [time, height, width], got {tuple(tactile.shape)}")
    if side <= 0:
        raise ValueError(f"side must be positive, got {side}")
    value = normalize(tactile.float().reshape(1, -1)).reshape_as(tactile)
    value = F.interpolate(value.unsqueeze(1), size=(side, side), mode="nearest")
    return value.expand(-1, 3, -1, -1)


def family_frames(signal: torch.Tensor, family: str, cell: int) -> torch.Tensor:
    """The family -> renderer dispatch shared by the ts-native builder and
    Stage-1 (``no duplicated math`` invariant, rl/ts_native_DESIGN.md)."""
    if family == "tactile":
        return tactile_frames(signal, cell)
    if family == "ecg":
        return timeseries_frames(signal, cell)
    raise ValueError(f"unsupported signal family {family!r}; expected one of {tuple(sorted(SIGNAL_FAMILIES))}")
