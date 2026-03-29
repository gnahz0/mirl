# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from io import BytesIO
from typing import Optional

_MAX_IMAGE_TOKENS = int(os.environ.get("QWEN_VL_MAX_IMAGE_TOKENS", "16384"))
_VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "0")) or None  # 0 or unset = no cap

# qwen_vl_utils: torchvision's read_video() decodes the entire clip, then subsamples to max_frames
# (slow / high RAM for long files). decord/torchcodec fetch only the sampled indices.
# Install decord: pip install decord. Override with FORCE_QWENVL_VIDEO_READER=torchvision|torchcodec|decord.
if "FORCE_QWENVL_VIDEO_READER" not in os.environ:
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
# Long or messy H.264/MP4 (e.g. web rips) can exceed decord's default EOF retries and fall back to
# full-file torchvision. Raise if you see "Unable to handle EOF ... DECORD_EOF_RETRY_MAX=1024".
if "DECORD_EOF_RETRY_MAX" not in os.environ:
    os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
# If 1/true: on string video paths, do not use qwen_vl_utils's torchvision fallback when the primary
# backend fails—raise so callers can skip the sample (see filter_by_token_limit.py setdefault).
_VERL_SKIP_QWENVL_TV_FALLBACK = os.environ.get(
    "VERL_SKIP_QWENVL_VIDEO_TORCHVISION_FALLBACK", "0"
).lower() in ("1", "true", "yes")

import torch
from PIL import Image, ImageFile

# Allow loading truncated/corrupt image files (common in medical/clinical datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    from qwen_vl_utils import fetch_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, dict) and "max_pixels" not in image and _MAX_IMAGE_TOKENS < 16384:
        image = dict(image)
        image["max_pixels"] = _MAX_IMAGE_TOKENS * image_patch_size * image_patch_size

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    try:
        ans = fetch_image(image, image_patch_size=image_patch_size)
    except Exception:
        ans = fetch_image(image)
    return ans


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def _qwen_video_string_path_no_torchvision_fallback(
    ele: dict,
    image_patch_size: int,
    return_video_sample_fps: bool,
    return_video_metadata: bool,
):
    """Same as qwen_vl_utils.fetch_video for string paths, but no torchvision fallback on read errors."""
    import qwen_vl_utils.vision_process as vp
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_factor = image_patch_size * vp.SPATIAL_MERGE_SIZE
    video_frame_min_pixels = vp.VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = vp.VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    backend = vp.get_video_reader_backend()
    try:
        video, video_metadata, sample_fps = vp.VIDEO_READER_BACKENDS[backend](ele)
    except Exception as e:
        raise RuntimeError(
            f"Video backend {backend!r} failed (torchvision fallback disabled via "
            f"VERL_SKIP_QWENVL_VIDEO_TORCHVISION_FALLBACK): {e}"
        ) from e

    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", video_frame_min_pixels)
    total_pixels = ele.get("total_pixels", vp.MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * vp.FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        vp.logger.warning(
            "The given max_pixels[%s] exceeds limit[%s].",
            max_pixels_supposed,
            max_pixels,
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = vp.smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = vp.smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
    max_frames_override: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """
    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)
    # JSON often has "min_frames": null / "max_frames": null. dict.get(k, default) still returns
    # None when the key is present, which breaks qwen_vl_utils.smart_nframes (ceil_by_factor).
    for _k in ("min_frames", "max_frames", "fps", "nframes"):
        if video.get(_k) is None:
            video.pop(_k, None)

    # Cap max_frames: explicit override > env VIDEO_MAX_FRAMES > no cap
    cap = max_frames_override if max_frames_override is not None else _VIDEO_MAX_FRAMES
    if cap is not None:
        video.setdefault("max_frames", cap)
        mf = video.get("max_frames", 768)
        if mf is not None and mf > cap:
            video["max_frames"] = cap

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    if _VERL_SKIP_QWENVL_TV_FALLBACK and isinstance(video.get("video"), str):
        return _qwen_video_string_path_no_torchvision_fallback(
            video,
            image_patch_size,
            return_video_sample_fps,
            return_video_metadata,
        )

    from qwen_vl_utils import fetch_video

    return fetch_video(
        video,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}
