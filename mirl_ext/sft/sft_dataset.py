"""MIRL-specific media normalization for the stock multi-turn SFT dataset."""

from __future__ import annotations

from functools import cached_property

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class MIRLSFTDataset(MultiTurnSFTDataset):
    """Stock position IDs plus MIRL's media-shape safeguards."""

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
