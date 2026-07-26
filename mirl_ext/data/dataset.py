"""MIRL's schema and bounded-media adapter for upstream ``RLHFDataset``."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from typing import Any

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


def _extra_info_as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


class MIRLDataset(RLHFDataset):
    """Normalize historical MIRL rows without patching upstream verl.

    MIRL parquet files intentionally use one schema across modality families.
    ``extra_info`` is a JSON string, nullable video fields are materialized as
    ``None``, and human-behaviour prompts contain an informational ``<audio>``
    marker although no raw audio column is consumed.  Upstream verl expects a
    dict, concrete video limits, and a real audio payload for that marker.
    """

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
                for field in ("min_frames", "max_frames", "fps", "nframes"):
                    if video.get(field) is None:
                        video.pop(field, None)
                if max_video_frames is not None:
                    video["max_frames"] = min(video.get("max_frames", 768), int(max_video_frames))
            videos.append(video)
        normalized[self.video_key] = videos
        # This class is loaded from a file path by verl. Hugging Face Dataset's
        # multiprocess filter serializes that dynamic class; zero-argument
        # ``super()`` can then resolve against a reconstructed class object.
        # Calling the stable imported base explicitly remains pickle-safe.
        return RLHFDataset._build_messages(self, normalized, key=key)

    def __getitem__(self, item):
        row: dict[str, Any] = self.dataframe[item]
        row["raw_prompt"] = self._build_messages(row, key=self.prompt_key)
        row.pop(self.image_key, None)
        row.pop(self.video_key, None)
        row.pop(self.audio_key, None)
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = _extra_info_as_dict(row.get("extra_info"))
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
                    for field in ("min_frames", "max_frames", "fps", "nframes"):
                        if item.get(field) is None:
                            item.pop(field, None)
                    if max_video_frames is not None:
                        item["max_frames"] = min(item.get("max_frames", 768), int(max_video_frames))
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
        from qwen_vl_utils import process_vision_info

        cls._prepare_media_messages(messages, image_patch_size=image_patch_size, config=config)
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") in {"image", "video"}
                for item in message["content"]
            )
            for message in messages
        )
        if has_visual:
            images, videos = process_vision_info(
                messages,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
        else:
            images, videos = None, None
        return images, videos, RLHFDataset._extract_audio_info(messages)

    @classmethod
    async def process_multi_modal_info(cls, messages: list[dict], image_patch_size, config):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls._process_multi_modal_info(messages, image_patch_size=image_patch_size, config=config),
        )
