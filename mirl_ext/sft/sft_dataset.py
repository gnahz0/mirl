"""Qwen3.5-aware SFT dataset (wired via data.custom_cls in the sbatch).

Qwen3.5's template renders a video as one timestamped ``video_token`` run per
temporal patch while the processor returns a single ``video_grid_thw=[T,H,W]``
row, so stock ``MultiTurnSFTDataset`` crashes in ``get_rope_index`` (it expects
one grid row per run). Same quirk the sanctioned agent_loop patch handles for
RL: expand the grid to one ``[1,H,W]`` row per run FOR POSITION IDS ONLY -- the
stored multi_modal_inputs keep the original grid the vision tower needs.
"""

from __future__ import annotations

from functools import cached_property

import torch
import verl.utils.dataset.multiturn_sft_dataset as _msd
from verl.models.transformers.qwen2_vl import get_rope_index as _get_rope_index
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def position_video_grid(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """[n,3] rows of [T,H,W] -> one [1,H,W] row per temporal patch run."""
    runs = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    runs[:, 0] = 1
    return runs


def _qwen35_rope_index(processor, input_ids, image_grid_thw=None, video_grid_thw=None,
                       second_per_grid_ts=None, attention_mask=None):
    if video_grid_thw is not None:
        video_grid_thw = position_video_grid(video_grid_thw)
        second_per_grid_ts = None  # timestamps already live in the text runs
    return _get_rope_index(processor, input_ids=input_ids, image_grid_thw=image_grid_thw,
                           video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts,
                           attention_mask=attention_mask)


# The rope call sits mid-__getitem__ with no hook; swapping the module symbol is
# the smallest override that avoids duplicating 130 lines of verl code.
_msd.get_rope_index = _qwen35_rope_index


class MIRLSFTDataset(MultiTurnSFTDataset):
    """Stock behavior + the position-id fix installed at import time."""

    # 1024 merged tokens/image (32px per token side), matching alignment's
    # max_image_tokens and the RL clamp; qwen_vl_utils' default ceiling (~12k
    # tokens) can blow max_length under truncation=error on multi-image rows.
    MAX_IMAGE_PIXELS = 1024 * 32 * 32

    @cached_property
    def modality_flags(self) -> list[bool]:
        """One video/image flag per dataframe row for synchronized DP batches."""
        images = self.dataframe[self.image_key].tolist() if self.image_key in self.dataframe else [None] * len(self)
        videos = self.dataframe[self.video_key].tolist() if self.video_key in self.dataframe else [None] * len(self)
        media_kinds = [(_has_items(image), _has_items(video)) for image, video in zip(images, videos, strict=True)]
        invalid = [i for i, kind in enumerate(media_kinds) if sum(kind) != 1]
        if invalid:
            preview = ", ".join(map(str, invalid[:8]))
            raise ValueError(
                "MIRL SFT requires exactly one media kind per row; "
                f"invalid dataframe positions: {preview}"
            )
        return [has_video for _, has_video in media_kinds]

    def _build_messages(self, example: dict):
        # pandas hands parquet list columns back as np.ndarray: never bool()
        # them (`x or []` raises on empty/multi-element arrays), and
        # qwen_vl_utils asserts list/tuple for frames-as-video entries.
        videos = example.get(self.video_key)
        for entry in [] if videos is None else videos:
            if isinstance(entry, dict) and not isinstance(entry.get("video"), (str, list, tuple)):
                entry["video"] = [str(p) for p in entry["video"]]
        images = example.get(self.image_key)
        for entry in [] if images is None else images:
            if isinstance(entry, dict):
                cap = entry.get("max_pixels")
                entry["max_pixels"] = min(int(cap), self.MAX_IMAGE_PIXELS) if cap else self.MAX_IMAGE_PIXELS
        return super()._build_messages(example)


def _has_items(value) -> bool:
    """Parquet list cells arrive as lists or numpy/Arrow arrays."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False
