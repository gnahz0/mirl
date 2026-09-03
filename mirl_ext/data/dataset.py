"""MIRL's schema and bounded-media adapter for upstream ``RLHFDataset``."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
from typing import Any

import torch
from PIL import Image

from mirl_ext.data.schema import HUMAN_BEHAVIOUR_SOURCES, MEDICAL_SOURCES, TACTILE_SOURCES
from mirl_ext.data.schema import extra_info as parse_extra_info
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

# ts-native strips (rl/build_ts_native_parquet.py): T Stage-1 pseudo-video
# frames stacked vertically in one grayscale PNG, frame count in the name.
_TS_STACK_RE = re.compile(r"_stack(\d+)\.png$")

# Match the media representations used for teacher-trace generation. SFT uses
# the teacher's frozen JPEG lists directly; these rules cover raw-video RL and
# evaluation rows from the disjoint RL/validation recordings.
TACTILE_VIDEO_FPS = 1.0
TACTILE_VIDEO_MIN_FRAMES = 4
TACTILE_VIDEO_MAX_FRAMES = 24
FIXED_VIDEO_FRAMES = 8


def _normalize_video(entry: dict, max_video_frames, data_source: str | None = None) -> None:
    """Sanitize qwen-vl sampling fields and apply MIRL's family policy.

    Tactile RGB+heatmap composites use approximately 1 FPS with a four-frame
    floor. Human-behavior uses exactly eight uniformly spaced frames. Other
    ordinary RGB videos (currently CLIMB) use at most eight because a few have
    fewer than eight source frames. Native TS stack pseudo-videos bypass
    qwen-vl sampling in ``_process_multi_modal_info``.
    """
    for field in ("min_frames", "max_frames", "fps", "nframes"):
        if entry.get(field) is None:
            entry.pop(field, None)

    source = entry.get("video")
    if isinstance(source, str) and _TS_STACK_RE.search(source):
        return

    if data_source is not None:
        if str(data_source) in TACTILE_SOURCES:
            entry.pop("nframes", None)
            entry.update(
                fps=TACTILE_VIDEO_FPS,
                min_frames=TACTILE_VIDEO_MIN_FRAMES,
                max_frames=TACTILE_VIDEO_MAX_FRAMES,
            )
        elif str(data_source) in HUMAN_BEHAVIOUR_SOURCES | MEDICAL_SOURCES:
            for field in ("fps", "min_frames", "max_frames"):
                entry.pop(field, None)
            entry["nframes"] = FIXED_VIDEO_FRAMES
        else:
            for field in ("fps", "min_frames", "max_frames", "nframes"):
                entry.pop(field, None)
            entry["max_frames"] = FIXED_VIDEO_FRAMES

    # qwen_vl_utils accepts either nframes or fps, never both. A fixed nframes
    # policy needs no separate max_frames field.
    if "nframes" in entry:
        for field in ("fps", "min_frames", "max_frames"):
            entry.pop(field, None)
    elif max_video_frames is not None:
        entry["max_frames"] = min(entry.get("max_frames", 768), int(max_video_frames))


def _relax_unavailable_fixed_frame_count(messages: list[dict], error: ValueError) -> int | None:
    """Set the largest supported even count for an exceptionally short video.

    qwen_vl_utils rejects explicit ``nframes=8`` when a video contains fewer
    than eight source frames. Parse its reported source-frame ceiling and retry
    with the closest count its two-frame temporal patching supports.
    """
    match = re.search(r"nframes should in interval \[2,\s*(\d+)\], but got 8", str(error))
    if match is None:
        return None
    candidates = [
        item
        for message in messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "video" and item.get("nframes") == FIXED_VIDEO_FRAMES
    ]
    if len(candidates) != 1:
        return None
    frame_count = min(FIXED_VIDEO_FRAMES, int(match.group(1)))
    frame_count -= frame_count % 2
    if frame_count < 2:
        return None
    candidates[0]["nframes"] = frame_count
    return frame_count


def fetch_ts_stack(path: str, nframes: int) -> tuple[torch.Tensor, dict]:
    """Strip PNG -> the ([T,3,H,W] float 0-255, metadata) tuple both the agent
    loop and the trainer already pass through for real videos.

    Deliberately bypasses qwen_vl_utils: its frame-list path feeds image_factor
    (32) to fetch_image as the PATCH size, so frames get smart_resized with
    factor 64 and the 32 px Stage-1 tile width doubles. The HF video processor
    is measured identity on these dims (rl/ts_native_DESIGN.md)."""
    import numpy as np

    strip = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    frames = torch.from_numpy(strip).reshape(nframes, -1, strip.shape[1])
    if nframes % 2:  # even frame count, same convention as qwen_vl_utils
        frames = torch.cat([frames, frames[-1:]])
        nframes += 1
    # Contiguous like fetch_video's output; stride-0 views can trip IPC paths.
    video = frames.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    metadata = {"fps": 2.0, "frames_indices": list(range(nframes)), "total_num_frames": float(nframes)}
    return video, metadata


class MIRLDataset(RLHFDataset):
    """Normalize MIRL parquet rows for upstream verl: ``extra_info`` arrives as
    a JSON string, video limit fields may be None, and human-behaviour prompts
    carry an ``<audio>`` marker with no audio payload."""

    def _build_messages(self, example: dict, key: str):
        normalized = copy.deepcopy(example)
        if not (normalized.get(self.audio_key, None) or []):
            for message in normalized[key]:
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = content.replace("<audio>\n", "").replace("<audio>", "")

        max_video_frames = self.config.get("max_video_frames")
        videos = []
        for raw_video in normalized.get(self.video_key, None) or []:
            video = dict(raw_video) if isinstance(raw_video, dict) else raw_video
            if isinstance(video, dict):
                _normalize_video(video, max_video_frames, normalized.get("data_source"))
            videos.append(video)
        normalized[self.video_key] = videos
        # Explicit base call, not super(): HF's multiprocess filter pickles this file-loaded class, and zero-arg super() can then resolve against a reconstructed class.
        return RLHFDataset._build_messages(self, normalized, key=key)

    def __getitem__(self, item):
        row: dict[str, Any] = self.dataframe[item]
        row["raw_prompt"] = self._build_messages(row, key=self.prompt_key)
        row.pop(self.image_key, None)
        row.pop(self.video_key, None)
        row.pop(self.audio_key, None)
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = parse_extra_info(row)
        row["extra_info"] = extra_info
        row["index"] = extra_info.get("index", 0)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        return row

    @classmethod
    def _prepare_media_messages(cls, messages: list[dict], image_patch_size: int, config) -> None:
        max_video_frames = config.get("max_video_frames") if config else None
        max_video_bytes = config.get("max_video_bytes") if config else None
        max_image_tokens = config.get("max_image_tokens") if config else None
        total_image_tokens = config.get("max_image_tokens_total") if config else None
        if max_image_tokens and not total_image_tokens:
            total_image_tokens = int(max_image_tokens) * 4

        image_count = sum(
            1
            for message in messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        )
        per_image_tokens = int(max_image_tokens) if max_image_tokens else None
        if per_image_tokens and total_image_tokens and image_count > 1:
            per_image_tokens = min(per_image_tokens, max(1, int(total_image_tokens) // image_count))
        max_image_pixels = per_image_tokens * image_patch_size**2 if per_image_tokens else None

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = []
            for item in content:
                if not isinstance(item, dict):
                    kept.append(item)
                    continue
                if item.get("type") == "image":
                    source = item.get("image")
                    if isinstance(source, str):
                        try:
                            with Image.open(source) as image:
                                image.verify()
                        except (OSError, ValueError) as exc:
                            logger.warning("Dropping unreadable image %s: %s", source, exc)
                            continue
                    if max_image_pixels is not None:
                        existing = item.get("max_pixels")
                        item["max_pixels"] = min(int(existing), max_image_pixels) if existing else max_image_pixels
                elif item.get("type") == "video":
                    source = item.get("video")
                    if isinstance(source, str) and _TS_STACK_RE.search(source):
                        # ts-native strip: fetched by _process_multi_modal_info,
                        # never by qwen_vl_utils; the caps below don't apply.
                        kept.append(item)
                        continue
                    _normalize_video(item, max_video_frames)
                    source = item.get("video")
                    if max_video_bytes and isinstance(source, str):
                        try:
                            if os.path.getsize(source) > int(max_video_bytes):
                                logger.warning("Dropping oversized video %s", source)
                                continue
                        except OSError:
                            pass
                kept.append(item)
            message["content"] = kept

    @classmethod
    def _process_multi_modal_info(cls, messages: list[dict], image_patch_size, config):
        cls._prepare_media_messages(messages, image_patch_size=image_patch_size, config=config)
        stacks, other_media = [], 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in ("image", "video"):
                    continue
                source = item.get("video")
                if isinstance(source, str) and (stack := _TS_STACK_RE.search(source)):
                    stacks.append(fetch_ts_stack(source, int(stack.group(1))))
                else:
                    other_media += 1
        if stacks:
            # ts-native rows carry exactly one media: the strip (builder invariant).
            assert not other_media and len(stacks) == 1, "ts-native strip must be the row's only media"
            return None, stacks, cls._extract_audio_info(messages)
        try:
            return RLHFDataset._process_multi_modal_info(
                messages,
                image_patch_size=image_patch_size,
                config=config,
            )
        except ValueError as error:
            fallback_frames = _relax_unavailable_fixed_frame_count(messages, error)
            if fallback_frames is None:
                raise
            logger.warning(
                "Video has fewer than %d source frames; retrying with %d",
                FIXED_VIDEO_FRAMES,
                fallback_frames,
            )
            return RLHFDataset._process_multi_modal_info(
                messages,
                image_patch_size=image_patch_size,
                config=config,
            )

    @classmethod
    async def process_multi_modal_info(cls, messages: list[dict], image_patch_size, config):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls._process_multi_modal_info(messages, image_patch_size=image_patch_size, config=config),
        )
