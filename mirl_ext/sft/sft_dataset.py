"""Qwen3.5-aware SFT dataset (wired via data.custom_cls in the sbatch).

Qwen3.5's template renders a video as one timestamped ``video_token`` run per
temporal patch while the processor returns a single ``video_grid_thw=[T,H,W]``
row, so stock ``MultiTurnSFTDataset`` crashes in ``get_rope_index`` (it expects
one grid row per run). Same quirk the sanctioned agent_loop patch handles for
RL: expand the grid to one ``[1,H,W]`` row per run FOR POSITION IDS ONLY -- the
stored multi_modal_inputs keep the original grid the vision tower needs.
"""

from __future__ import annotations

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
