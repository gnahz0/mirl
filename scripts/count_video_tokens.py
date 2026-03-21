#!/usr/bin/env python3
"""Count prompt tokens for HB video samples using Qwen3-VL processor."""
import json
import os
import sys

COMBINED_TRAIN_DEMO_JSON = "/home/alecz/mirl/data/combined_train_demo_only.json"


def main():
    from transformers import AutoProcessor
    from verl.utils.dataset.vision_utils import process_video

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # Find HB samples with video
    hb_with_video = []
    with open(COMBINED_TRAIN_DEMO_JSON) as f:
        for line in f:
            e = json.loads(line)
            if e.get("videos") and e.get("data_source") in ("ptsd_in_the_wild", "mosei_emotion", "meld_emotion"):
                hb_with_video.append(e)
                if len(hb_with_video) >= 5:
                    break

    print(f"Testing {len(hb_with_video)} HB video samples\n")

    for i, sample in enumerate(hb_with_video):
        ds = sample["data_source"]
        prompt = sample["prompt"]
        videos = sample["videos"]

        # Check if video file exists
        vid_path = videos[0].get("video") if isinstance(videos[0], dict) else videos[0]
        exists = os.path.isfile(vid_path)
        print(f"[{i}] {ds}: video={vid_path[:60]}... exists={exists}")

        if not exists:
            # Estimate: 8 frames, ~256 tokens/frame at 448x448 = ~2048 video tokens
            # Qwen-VL: each frame similar to image, patch_size=14, typical ~256-512 tokens/frame
            est = 8 * 256
            print(f"     (file missing, estimate: 8 frames * ~256 tok/frame = ~{est} video tokens)")
            continue

        try:
            videos_processed, video_metadata = zip(
                *[
                    process_video(
                        v, image_patch_size=14, return_video_metadata=True
                    )
                    for v in videos
                ],
                strict=True,
            )
            videos_kwargs = {"video_metadata": list(video_metadata), "do_sample_frames": False}

            # Build content like rl_dataset
            raw_parts = []
            for m in prompt:
                c = m.get("content", "")
                # Replace <video> with placeholder for processor
                c = c.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
                raw_parts.append(c)
            raw_prompt = "\n".join(raw_parts)

            out = proc(
                text=[raw_prompt],
                images=None,
                videos=list(videos_processed),
                videos_kwargs=videos_kwargs,
            )
            n_tokens = len(out["input_ids"][0])
            n_video_tokens = (out["input_ids"][0] == proc.video_token_id).sum().item()
            print(f"     total_tokens={n_tokens}, video_placeholder_tokens={n_video_tokens}")
        except Exception as ex:
            print(f"     ERROR: {ex}")

if __name__ == "__main__":
    main()
