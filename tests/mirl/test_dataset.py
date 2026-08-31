from __future__ import annotations

import asyncio
import json

from omegaconf import OmegaConf
from PIL import Image

from mirl_ext.data.dataset import MIRLDataset


def _bare_dataset(config=None):
    dataset = object.__new__(MIRLDataset)
    dataset.audio_key = "audios"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.prompt_key = "prompt"
    dataset.need_tools_kwargs = False
    dataset.processor = object()
    dataset.config = OmegaConf.create(config or {"max_video_frames": 8})
    return dataset


def test_schema_normalization_handles_json_extra_info_and_phantom_audio():
    dataset = _bare_dataset()
    dataset.dataframe = [
        {
            "data_source": "ptsd_in_the_wild",
            "prompt": [{"role": "user", "content": "<video>\n<audio>\ntranscript"}],
            "images": [],
            "videos": [{"video": "/scratch/example/clip.mp4", "min_frames": None, "max_frames": 12}],
            "reward_model": {"ground_truth": "no ptsd", "style": "rule"},
            "extra_info": json.dumps({"index": 7, "dataset": "ptsd"}),
        }
    ]

    row = dataset[0]
    assert row["index"] == 7
    assert row["extra_info"]["dataset"] == "ptsd"
    content = row["raw_prompt"][0]["content"]
    assert [item["type"] for item in content] == ["video", "text"]
    assert content[0]["max_frames"] == 8
    assert "<audio>" not in content[1]["text"]


def test_ts_stack_bypasses_qwen_vl_utils(tmp_path):
    """A _stack{T}.png video must come back as the raw (tensor, metadata) tuple:
    32 px tile width preserved (qwen_vl_utils would resize it to 64), odd frame
    count padded by repeating the last frame."""
    import numpy as np
    import torch

    pixels = np.random.default_rng(0).integers(0, 256, (5 * 64, 32), dtype=np.uint8)
    strip = tmp_path / "abc123_stack5.png"
    Image.fromarray(pixels, "L").save(strip)
    messages = [{"role": "user", "content": [{"type": "video", "video": str(strip)}]}]

    images, videos, audios = asyncio.run(
        MIRLDataset.process_multi_modal_info(
            messages, image_patch_size=16, config=OmegaConf.create({"max_video_frames": 8})
        )
    )
    assert images is None and audios is None and len(videos) == 1
    video, metadata = videos[0]
    assert tuple(video.shape) == (6, 3, 64, 32)
    assert torch.equal(video[5], video[4])
    assert video[0, 0].numpy().tolist() == pixels[:64].tolist()
    assert metadata["fps"] == 2.0 and metadata["total_num_frames"] == 6.0


def test_image_budget_is_applied_during_async_rollout(tmp_path):
    image_path = tmp_path / "large.png"
    Image.new("RGB", (1024, 1024), color="white").save(image_path)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": str(image_path)}]}
    ]
    config = OmegaConf.create({"max_image_tokens": 64, "max_image_tokens_total": 64})

    images, videos, audios = asyncio.run(
        MIRLDataset.process_multi_modal_info(messages, image_patch_size=16, config=config)
    )
    assert len(images) == 1
    assert images[0].size == (128, 128)
    assert videos is None
    assert audios is None
