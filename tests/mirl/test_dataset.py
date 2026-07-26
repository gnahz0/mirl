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
