# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Dataset wrapper for Stage 1 multimodal alignment.

Reuses the repo's existing Parquet/JSONL row schema (no schema changes):
    data_source: str
    prompt: [{role, content}]               (content has <image>/<video>/<audio> placeholders)
    images: [{"image": "/abs/path.png", ...}]
    videos: [{"video": "...mp4", "max_frames": int}]
    audios: [{"audio": "...wav"}]
    reward_model: {"style": "rule", "ground_truth": str}
    extra_info: dict or JSON string

Routing logic:
    * A non-empty ``signals`` field -> ``branch="ts"``. The loader dispatches
      SmellNet CSV, ECG tensor, and haptic dictionary formats explicitly.
    * Otherwise -> ``branch="img"``.
    * For the img branch: use the first image (``media_kind="image"``) else the first
      video frame-tensor (``media_kind="video"``, decoded via
      ``verl/utils/dataset/vision_utils.process_video``).
    * Else                                                       -> dropped.

New schema field (ts branch only)::

    signals: [{"signal": "/abs/path.csv"}]

Audio inputs are still TODO(stage2). Stage 1 only handles visual + raw-signal modalities.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import signal
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Match scripts/diagnose_skipped_videos.py: torchcodec is the preferred backend
# (avoids decord hangs on some MP4s) and we want it on in *all* processes including
# DataLoader workers, so set it at import time before qwen_vl_utils is touched.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("TORCHCODEC_LOG_LEVEL", "0")


_PLACEHOLDER_RE = re.compile(r"<image>|<video>|<audio>")


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(" ", text).strip()


def _user_question(prompt: list[dict]) -> str:
    """Last user-turn text content with <image>/<video>/<audio> placeholders removed."""
    for msg in reversed(prompt or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return _strip_placeholders(content)
            if isinstance(content, list):
                parts = []
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(it.get("text", ""))
                return _strip_placeholders(" ".join(parts))
    return ""


def _text_for_label_encoder(sample: dict, mode: str) -> str:
    gt = (sample.get("reward_model") or {}).get("ground_truth", "")
    if isinstance(gt, list):
        gt = ", ".join(str(x) for x in gt)
    gt = str(gt or "").strip()
    if mode == "ground_truth":
        return gt
    if mode == "question":
        return _user_question(sample.get("prompt") or [])
    if mode == "question_plus_gt":
        q = _user_question(sample.get("prompt") or [])
        return f"{q} | {gt}".strip(" |")
    raise ValueError(f"unknown text_for_label mode {mode!r}")


def _load_image_path_or_dict(img_entry) -> Optional[Image.Image]:
    """Lazy, robust image load. Returns ``None`` on any failure so the sample can be dropped."""
    try:
        from verl.utils.dataset.vision_utils import process_image
        return process_image(img_entry)
    except Exception:  # noqa: BLE001
        try:
            if isinstance(img_entry, Image.Image):
                return img_entry.convert("RGB")
            if isinstance(img_entry, dict):
                path = img_entry.get("image") or img_entry.get("path")
                if path and os.path.exists(path):
                    return Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    return None


_TIME_COL_HINTS = ("timestamp", "time_ms", "time")


def _signal_path(sig_entry) -> Optional[str]:
    if isinstance(sig_entry, str):
        return sig_entry
    if isinstance(sig_entry, dict):
        return sig_entry.get("signal") or sig_entry.get("path") or sig_entry.get("csv")
    return None


def _load_signal_csv(
    sig_entry,
    in_channels: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Load a raw multivariate CSV into a native-shape ``(channels, time)`` tensor.

    * Drops any timestamp/time column (detected by header name).
    * Optionally pads/truncates channels or time for backwards-compatible experiments.
      The Qwen3.5 alignment configs leave both unset so 4-channel mixtures, 6-channel
      base samples, and every native sequence length remain intact.

    Returns ``None`` on any failure so the sample is dropped.
    """
    path = _signal_path(sig_entry)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            header = f.readline().strip().split(",")
        keep_idx = [
            i for i, name in enumerate(header)
            if not any(h in name.strip().lower() for h in _TIME_COL_HINTS)
        ]
        if not keep_idx:
            return None
        data = np.genfromtxt(
            path, delimiter=",", skip_header=1, usecols=keep_idx, dtype=np.float32,
        )
        if data.ndim == 1:
            data = data[:, None] if len(keep_idx) == 1 else data[None, :]
        # data is (T, C); transpose to (C, T).
        arr = np.asarray(data, dtype=np.float32).T

        c, t = arr.shape
        # Channel pad / truncate.
        if in_channels and c < in_channels:
            arr = np.pad(arr, ((0, in_channels - c), (0, 0)), mode="constant")
        elif in_channels and c > in_channels:
            arr = arr[:in_channels]
        # Time pad (edge-replicate last column) / truncate.
        if seq_len and t < seq_len:
            pad = np.repeat(arr[:, -1:], seq_len - t, axis=1) if t > 0 \
                else np.zeros((arr.shape[0], seq_len), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif seq_len and t > seq_len:
            arr = arr[:, :seq_len]
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    except Exception as e:  # noqa: BLE001
        logger.debug("signal load failed (%s): %s", type(e).__name__, str(e)[:160])
        return None


def _load_signal_pt_1d(path: str, target_len: Optional[int] = None) -> Optional[torch.Tensor]:
    """Load a 1-D multivariate ``(C, T)`` signal saved as a ``.pt`` tensor (e.g. ECG 8x2500).

    Optionally resamples the time axis to ``target_len`` (linear interp -- preserves the
    whole trace, unlike truncation). Returns ``None`` on failure so the sample is dropped.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            for key in ("signal", "data", "ts"):
                if key in t and t[key] is not None:
                    t = t[key]
                    break
            else:
                return None
        t = torch.as_tensor(t).float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if target_len and t.shape[1] != target_len and t.shape[1] > 1:
            t = F.interpolate(t.unsqueeze(0), size=int(target_len),
                              mode="linear", align_corners=False).squeeze(0)
        return t.contiguous()
    except Exception as e:  # noqa: BLE001
        logger.debug("pt 1d load failed (%s): %s", type(e).__name__, str(e)[:160])
        return None


def _load_tactile_pt(
    path: str, key: str = "glove_right", max_frames: Optional[int] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Load right tactile maps and aligned right-hand force statistics.

    v1 stores one right tactile map and 13 right-force columns. v2 additionally
    stores left/aligned/mat maps and 26 force columns; for a schema consistent with
    v1 and the requested ``glove_right`` signal, only the 13 ``right_*`` columns
    accompany the selected right map. One known v1 record has no force statistics;
    it receives an empty force tensor and remains usable.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        tac = (d.get("tactile") or {}).get(key) if isinstance(d, dict) else d
        if tac is None:
            return None
        t = torch.as_tensor(tac).float()
        if t.numel() == 0:
            return None
        if t.ndim == 2:
            t = t.unsqueeze(0)
        force = None
        columns = d.get("hand_force_columns") if isinstance(d, dict) else None
        force_raw = d.get("hand_force_stats") if isinstance(d, dict) else None
        if force_raw is not None and columns:
            force_all = torch.as_tensor(force_raw).float()
            right_idx = [i for i, name in enumerate(columns) if str(name).startswith("right_")]
            if right_idx:
                force = force_all[:, right_idx]
                if force.shape[0] != t.shape[0]:
                    if force.shape[0] > 1:
                        force = F.interpolate(
                            force.t().unsqueeze(0),
                            size=t.shape[0],
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0).t()
                    elif force.shape[0] == 1:
                        force = force.expand(t.shape[0], -1)
        if force is None:
            force = t.new_empty((t.shape[0], 0))

        if max_frames and t.shape[0] > max_frames:
            idx = torch.linspace(0, t.shape[0] - 1, int(max_frames)).round().long()
            t = t[idx]
            force = force[idx]
        return {"tactile": t.contiguous(), "force": force.contiguous()}
    except Exception as e:  # noqa: BLE001
        logger.debug("tactile load failed (%s): %s", type(e).__name__, str(e)[:160])
        return None


class _VideoTimeout(Exception):
    pass


def _video_timeout_handler(signum, frame):  # noqa: ARG001
    raise _VideoTimeout()


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Redirect file descriptor 2 to /dev/null while inside this context.

    Catches *C-level* writes from torchcodec / ffmpeg (``Could not open input file...``)
    that Python-level logging silencing can't reach. Falls through silently if the
    fd dance isn't possible (some sandboxed environments).
    """
    try:
        old_fd = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(old_fd, 2)
        finally:
            os.close(devnull)
            os.close(old_fd)


def _load_video_entry(
    video_entry: dict,
    image_patch_size: int = 14,
    max_frames_override: Optional[int] = None,
    timeout_sec: int = 30,
    suppress_stderr: bool = True,
):
    """Returns ``(video_tensor [n_frames, 3, H, W], video_metadata)`` or ``None``.

    Defensive wrapper around ``verl/utils/dataset/vision_utils.process_video``:

    * Sets a SIGALRM timeout (default 30s) so a malformed MP4 cannot hang the
      DataLoader. Matches the pattern from ``scripts/diagnose_skipped_videos.py``.
    * Suppresses C-level stderr from torchcodec / ffmpeg by default so unreadable
      files don't flood the console (the dataset already drops them silently).
    * Falls back to a plain ``process_video`` call if SIGALRM isn't usable in this
      thread / OS.

    Returns ``None`` on any failure (timeout, missing file, decode error) so the
    caller can route the sample to ``branch="none"`` and the collator drops it.
    """
    from verl.utils.dataset.vision_utils import process_video

    old_handler = None
    alarm_set = False
    try:
        try:
            old_handler = signal.signal(signal.SIGALRM, _video_timeout_handler)
            signal.alarm(int(timeout_sec))
            alarm_set = True
        except (ValueError, OSError):
            # SIGALRM unavailable (non-main thread, Windows, etc.); continue without timeout.
            pass

        ctx = _suppress_fd_stderr() if suppress_stderr else contextlib.nullcontext()
        with ctx:
            return process_video(
                video_entry,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
                max_frames_override=max_frames_override,
            )
    except _VideoTimeout:
        logger.warning("video decode timed out after %ds: %s", timeout_sec, video_entry)
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("video decode failed (%s): %s", type(e).__name__, str(e)[:160])
        return None
    finally:
        if alarm_set:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass


class AlignmentDataset(Dataset):
    """Yields per-sample dicts whose media is loaded lazily on ``__getitem__``.

    Each entry has::

        {
            "branch":      "img" | "ts" | "none",
            "media_kind":  "image" | "video" | "none",
            "media":       PIL.Image  OR  (video_tensor, video_metadata)  OR  None,
            "text":        str,            # text fed to the frozen label encoder
            "data_source": str,
            "index":       int,
        }

    Samples with ``branch == "none"`` are dropped by the collator.
    """

    def __init__(
        self,
        data_files: str | list[str],
        ts_data_sources: list[str],
        text_for_label: str = "ground_truth",
        max_samples: int = -1,
        balanced_sampling_key: Optional[str] = None,
        seed: Optional[int] = 42,
        enable_videos: bool = True,
        max_video_frames: Optional[int] = 8,
        image_patch_size: int = 14,
        video_load_timeout: int = 30,
        video_suppress_stderr: bool = True,
        data_source_filter: Optional[list[str]] = None,
        ts_in_channels: Optional[int] = None,
        ts_seq_len: Optional[int] = None,
        ts_oversample: int = 1,
        ts_pt_target_len: Optional[int] = None,
        tactile_max_frames: Optional[int] = None,
        include_all_ts: bool = False,
        max_img_samples: int = -1,
    ):
        """
        ...
        data_source_filter:
            Optional whitelist of ``data_source`` values to keep. When set, all rows
            whose ``data_source`` is not in this list are dropped *before* sampling.
            Useful for smoke-testing a specific branch (e.g. force ts samples by
            filtering to ``[smellnet_base, smellnet_mixture]``).
        ts_oversample:
            Integer replication factor for ts-source rows (default 1 = no change).
            smellnet is only ~1.9k rows, so balanced sampling already grabs all of them
            but they're still a tiny slice of each batch. Setting e.g. 10 makes each ts
            row appear 10x per epoch, raising the ts share of every batch. NOTE: this
            repeats the SAME ~1.9k signals, so high factors increase overfitting risk on
            the ts path -- it boosts batch *representation*, not data diversity.
        """
        if isinstance(data_files, str):
            data_files = [data_files]
        self.ts_sources = set(ts_data_sources or [])
        self.data_source_filter = set(data_source_filter) if data_source_filter else None
        self.text_for_label_mode = text_for_label
        self.seed = seed
        self.enable_videos = enable_videos
        self.max_video_frames = max_video_frames
        self.image_patch_size = image_patch_size
        self.video_load_timeout = int(video_load_timeout)
        self.video_suppress_stderr = bool(video_suppress_stderr)
        self.ts_in_channels = int(ts_in_channels) if ts_in_channels else None
        self.ts_seq_len = int(ts_seq_len) if ts_seq_len else None
        self.ts_pt_target_len = int(ts_pt_target_len) if ts_pt_target_len else None
        self.tactile_max_frames = int(tactile_max_frames) if tactile_max_frames else None

        rows: list[dict] = []
        for path in data_files:
            t0 = time.time()
            size_mb = (os.path.getsize(path) / (1024 * 1024)) if os.path.exists(path) else -1
            logger.info("loading %s (%.1f MB)", path, size_mb)
            n_before = len(rows)
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                rows.extend(pq.read_table(path).to_pylist())
            elif path.endswith(".jsonl") or path.endswith(".json"):
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                raise ValueError(f"unsupported data file: {path}")
            logger.info("  -> %d rows in %.1fs", len(rows) - n_before, time.time() - t0)

        if self.data_source_filter is not None:
            before = len(rows)
            rows = [r for r in rows if r.get("data_source") in self.data_source_filter]
            logger.info("data_source_filter=%s: %d -> %d rows",
                        sorted(self.data_source_filter), before, len(rows))

        def _has_media(r: dict) -> bool:
            # Content-based routing: a row with a non-empty ``signals`` field is a ts
            # (raw-signal) sample regardless of data_source; otherwise it's img/video.
            if r.get("signals"):
                return True
            if r.get("images"):
                return True
            if self.enable_videos and r.get("videos"):
                return True
            return False

        rows = [r for r in rows if _has_media(r)]
        if not rows:
            raise RuntimeError(
                f"no image/video-bearing samples found in {data_files} "
                f"(enable_videos={self.enable_videos})"
            )

        if include_all_ts:
            # Keep every ts (raw-signal) row; stratify-cap only the img/video rows. Lets us
            # use the full ts pool (e.g. all ~79k ECG) while bounding the much larger
            # image/video pool to a balanced budget.
            ts_rows = [r for r in rows if r.get("signals")]
            img_rows = [r for r in rows if not r.get("signals")]
            if max_img_samples and max_img_samples > 0:
                img_rows = self._maybe_stratified_sample(
                    img_rows, max_img_samples, balanced_sampling_key
                )
            rows = ts_rows + img_rows
            random.Random(self.seed).shuffle(rows)
            logger.info("include_all_ts: kept %d ts + %d img/video rows (max_img_samples=%s)",
                        len(ts_rows), len(img_rows), max_img_samples)
        elif max_samples and max_samples > 0:
            rows = self._maybe_stratified_sample(rows, max_samples, balanced_sampling_key)

        # Oversample ts rows (replicate) to raise their share of each batch. Done AFTER
        # sampling so it operates on the rows actually selected for this run.
        ts_oversample = int(ts_oversample)
        if ts_oversample > 1 and self.ts_sources:
            ts_rows = [r for r in rows if r.get("data_source") in self.ts_sources]
            if ts_rows:
                rows = rows + ts_rows * (ts_oversample - 1)
                random.Random(self.seed).shuffle(rows)
                logger.info(
                    "ts_oversample=%d: replicated %d ts rows -> %d total rows (ts share ~%.1f%%)",
                    ts_oversample, len(ts_rows), len(rows),
                    100.0 * len(ts_rows) * ts_oversample / max(1, len(rows)),
                )

        self.rows = rows
        logger.info(
            "AlignmentDataset: %d rows from %d files (videos=%s, max_video_frames=%s)",
            len(self.rows), len(data_files), self.enable_videos, self.max_video_frames,
        )

    def _maybe_stratified_sample(
        self, rows: list[dict], n: int, key: Optional[str]
    ) -> list[dict]:
        """Sample ~``n`` rows balanced across ``key`` buckets, water-filling to hit ``n``.

        Each bucket gets an equal share; buckets smaller than their share are taken whole
        and their leftover quota is redistributed (round by round) to buckets that still
        have rows. This reaches the full target ``n`` even when bucket sizes are very
        uneven (e.g. climb has millions, tactile only ~28k) -- a plain ``n // n_groups``
        split would otherwise undershoot badly.
        """
        rng = random.Random(self.seed)
        if not key:
            rng.shuffle(rows)
            return rows[:n]
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key, "__unknown__")), []).append(r)
        order = sorted(buckets)
        for k in order:
            rng.shuffle(buckets[k])
        total = sum(len(v) for v in buckets.values())
        n = min(n, total)
        pos = {k: 0 for k in order}
        out: list[dict] = []
        remaining = n
        while remaining > 0:
            active = [k for k in order if pos[k] < len(buckets[k])]
            if not active:
                break
            share = max(1, remaining // len(active))
            progressed = False
            for k in active:
                if remaining <= 0:
                    break
                take = min(share, len(buckets[k]) - pos[k], remaining)
                if take <= 0:
                    continue
                out.extend(buckets[k][pos[k]:pos[k] + take])
                pos[k] += take
                remaining -= take
                progressed = True
            if not progressed:
                break
        rng.shuffle(out)
        logger.info("stratified sample: %d from %d (%d groups by %r)",
                    len(out), total, len(order), key)
        return out[:n]

    def __len__(self) -> int:
        return len(self.rows)

    def _none_item(self, idx: int, ds: str) -> dict:
        return {
            "branch": "none", "media_kind": "none", "media": None,
            "text": "", "data_source": ds, "index": idx,
        }

    def _load_signal_any(self, sig_entry) -> tuple[Optional[torch.Tensor], str]:
        """Dispatch on the signal ``format`` tag. Returns ``(tensor, ts_format)``.

        ``ts_format`` is ``"smell"``, ``"ecg"``, or ``"tactile"``.
        """
        fmt = sig_entry.get("format") if isinstance(sig_entry, dict) else None
        if fmt == "ts_pt":
            return _load_signal_pt_1d(_signal_path(sig_entry), self.ts_pt_target_len), "ecg"
        if fmt == "tactile_pt":
            key = sig_entry.get("key", "glove_right") if isinstance(sig_entry, dict) else "glove_right"
            return _load_tactile_pt(_signal_path(sig_entry), key, self.tactile_max_frames), "tactile"
        # Default: SmellNet CSV. Native channel count and length are preserved unless
        # the optional legacy caps were explicitly configured.
        sig = _load_signal_csv(
            sig_entry, in_channels=self.ts_in_channels, seq_len=self.ts_seq_len,
        )
        return sig, "smell"

    def __getitem__(self, idx: int) -> dict:
        sample = self.rows[idx]
        ds = sample.get("data_source", "")
        # Content-based routing: any row carrying signals -> ts branch.
        branch = "ts" if sample.get("signals") else "img"

        # ts branch: load the RAW signal (no rendered plot). media_kind="signal".
        if branch == "ts":
            signals_meta = sample.get("signals") or []
            if signals_meta:
                sig, ts_format = self._load_signal_any(signals_meta[0])
                if sig is not None:
                    return {
                        "branch": "ts", "media_kind": "signal", "media": sig,
                        "ts_format": ts_format,
                        "text": _text_for_label_encoder(sample, self.text_for_label_mode),
                        "data_source": ds, "index": idx,
                    }
            return self._none_item(idx, ds)

        # 1. Prefer images (cheaper, deterministic). img-branch text is unused
        #    (images only feed the distillation loss), so it stays empty.
        images_meta = sample.get("images") or []
        if images_meta:
            pil = _load_image_path_or_dict(images_meta[0])
            if pil is not None:
                return {
                    "branch": branch, "media_kind": "image", "media": pil,
                    "text": "",
                    "data_source": ds, "index": idx,
                }

        # 2. Fall back to videos.
        # Tactile / HB / CLIMB video subsets feed the preservation branch.
        videos_meta = sample.get("videos") or []
        if self.enable_videos and videos_meta:
            loaded = _load_video_entry(
                videos_meta[0],
                image_patch_size=self.image_patch_size,
                max_frames_override=self.max_video_frames,
                timeout_sec=self.video_load_timeout,
                suppress_stderr=self.video_suppress_stderr,
            )
            if loaded is not None:
                return {
                    "branch": branch, "media_kind": "video", "media": loaded,
                    "text": "",
                    "data_source": ds, "index": idx,
                }

        return self._none_item(idx, ds)


def collate_alignment(batch: list[dict]) -> dict:
    """Split per-sample dicts into branch buckets.

    Returned dict::

        {
          # img branch (normal images / video frames; used for distillation only)
          "img_image_pil":  [PIL.Image, ...],
          "img_video":      [(tensor, meta), ...],
          # ts branch (raw signals -> pseudo-images; used for the contrastive loss)
          "ts_signal":      [Tensor|dict, ...],  # native 1-D tensor or tactile+force payload
          "ts_format":      [str, ...],      # parallel: "smell" | "ecg" | "tactile"
          "ts_signal_text": [str, ...],
          "img_meta":       [{data_source, index, kind}, ...],  # (images, then videos)
          "ts_meta":        [...]
        }
    """
    out = {
        "img_image_pil": [],
        "img_video":     [],
        "ts_signal":     [], "ts_format": [], "ts_signal_text": [],
        "img_meta":      [], "ts_meta":        [],
    }
    # Single pass: bucket each item by (branch, media_kind). img_meta must list
    # images before videos (the encoder concatenates them in that order), so
    # videos go to a side list and are appended after.
    video_meta: list[dict] = []
    for item in batch:
        kind, media = item["media_kind"], item["media"]
        if media is None:
            continue
        meta = {"data_source": item["data_source"], "index": item["index"], "kind": kind}
        if item["branch"] == "img" and kind == "image":
            out["img_image_pil"].append(media)
            out["img_meta"].append(meta)
        elif item["branch"] == "img" and kind == "video":
            out["img_video"].append(media)
            video_meta.append(meta)
        elif item["branch"] == "ts" and kind == "signal":
            # ts signals keep their native (variable) shape -- the model builds per-sample
            # pseudo-images and concatenates them (no fixed-shape stacking here).
            out["ts_signal"].append(media)
            out["ts_format"].append(item.get("ts_format", "smell"))
            out["ts_signal_text"].append(item["text"])
            out["ts_meta"].append(meta)
    out["img_meta"].extend(video_meta)
    return out
