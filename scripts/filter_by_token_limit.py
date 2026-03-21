"""Filter dataset entries by prompt token length.

Reads JSONL from data/, computes token length (including images/videos) with
Qwen3-VL processor, keeps only entries <= max_tokens, writes to new files.

Does not overwrite existing files; outputs to *_filtered_<N>.json

Usage:
  python scripts/filter_by_token_limit.py --max-tokens 4096 --data-dir data

  # Only check HB entries (CLIMB passes through):
  python scripts/filter_by_token_limit.py --only-check-hb --data-dir data

  # Specify files explicitly:
  python scripts/filter_by_token_limit.py \\
      --inputs data/combined_train_demo_only.json data/combined_valid_demo_only.json \\
      --max-tokens 4096 \\
      --data-dir data
"""

import argparse
import copy
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoProcessor

# Add project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.dataset.vision_utils import process_image, process_video


def build_messages(example: dict, prompt_key: str, image_key: str, video_key: str, data_source_dir: str):
    """Replace <image> and <video> placeholders in messages. Returns new messages list."""
    messages = list(example[prompt_key])
    images = list(example.get(image_key) or [])
    videos = list(example.get(video_key) or [])

    new_messages = []
    image_offset, video_offset = 0, 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            new_messages.append(dict(msg))
            continue

        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [s for s in segments if s]
        for segment in segments:
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(f"Missing image at offset {image_offset}")
                image = images[image_offset]
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    content_list.append({"type": "image", "image": image})
                elif isinstance(image, dict):
                    img = dict(image)
                    if "image" in img and not os.path.isabs(str(img.get("image", ""))):
                        img["image"] = os.path.join(data_source_dir, img["image"])
                    content_list.append({"type": "image", **img})
                else:
                    content_list.append({"type": "image", "image": image})
                image_offset += 1
            elif segment == "<video>":
                if video_offset >= len(videos):
                    raise ValueError(f"Missing video at offset {video_offset}")
                video = dict(videos[video_offset]) if isinstance(videos[video_offset], dict) else {"video": videos[video_offset]}
                if "video" in video and not os.path.isabs(str(video.get("video", ""))):
                    video["video"] = os.path.join(data_source_dir, video["video"])
                content_list.append({"type": "video", **video})
                video_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        new_messages.append({"role": msg["role"], "content": content_list})
    return new_messages


def compute_prompt_length(
    doc: dict,
    processor,
    prompt_key: str = "prompt",
    image_key: str = "images",
    video_key: str = "videos",
    data_source_dir: str = ".",
    image_patch_size: int = 14,
    max_video_frames: Optional[int] = None,
) -> int:
    """Compute token length for one entry. Uses a copy to avoid mutating doc."""
    doc = copy.deepcopy(doc)
    images_raw = list(doc.get(image_key) or [])
    videos_raw = list(doc.get(video_key) or [])

    messages = build_messages(doc, prompt_key, image_key, video_key, data_source_dir)

    raw_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    images = None
    if images_raw:
        images = [
            process_image(img, image_patch_size=image_patch_size) for img in images_raw
        ]

    videos = None
    videos_kwargs = {}
    if videos_raw:
        # Explicitly cap max_frames per video (don't rely on env - workers may not inherit)
        capped = []
        for v in videos_raw:
            vc = dict(v) if isinstance(v, dict) else {"video": v}
            if max_video_frames is not None:
                vc["max_frames"] = min(vc.get("max_frames", 768), max_video_frames)
            capped.append(vc)
        vs, metas = zip(
            *[
                process_video(v, image_patch_size=image_patch_size, return_video_metadata=True)
                for v in capped
            ],
            strict=True,
        )
        videos = list(vs)
        videos_kwargs = {"video_metadata": list(metas), "do_sample_frames": False}

    inputs = processor(
        text=[raw_prompt],
        images=images,
        videos=videos,
        videos_kwargs=videos_kwargs,
    )
    return len(inputs["input_ids"][0])


def try_truncate_entry(
    entry: dict,
    processor,
    max_tokens: int,
    data_source_dir: str,
    max_video_frames: Optional[int],
    prompt_key: str = "prompt",
    image_key: str = "images",
    video_key: str = "videos",
) -> Optional[tuple[int, dict]]:
    """Try to truncate an overlong entry to fit. Returns (token_count, truncated_entry) or None."""
    entry = copy.deepcopy(entry)
    messages = list(entry[prompt_key])

    def _length(e):
        n = compute_prompt_length(e, processor, data_source_dir=data_source_dir,
                                  max_video_frames=max_video_frames, prompt_key=prompt_key,
                                  image_key=image_key, video_key=video_key)
        return n

    n = _length(entry)
    if n <= max_tokens:
        return (n, entry)

    # 1. Shorten system prompt
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            if isinstance(msg.get("content"), str):
                messages[i] = {**msg, "content": "You are a helpful assistant."}
            elif isinstance(msg.get("content"), list):
                # multimodal system - replace text segments
                new_content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        new_content.append({"type": "text", "text": "You are a helpful assistant."})
                    else:
                        new_content.append(item)
                messages[i] = {**msg, "content": new_content}
            entry = copy.deepcopy(entry)
            entry[prompt_key] = messages
            n = _length(entry)
            if n <= max_tokens:
                return (n, entry)
            break

    # 2. Remove system message entirely (keep user/assistant only)
    messages_no_sys = [m for m in messages if m.get("role") != "system"]
    if messages_no_sys and len(messages_no_sys) < len(messages):
        entry = copy.deepcopy(entry)
        entry[prompt_key] = messages_no_sys
        n = _length(entry)
        if n <= max_tokens:
            return (n, entry)

    # 3. For multi-turn, drop oldest conversation turns (keep system + last user)
    while len(messages) > 2:
        # Drop first user + assistant pair; keep system if present
        if messages[0].get("role") == "system":
            messages = [messages[0]] + messages[3:]  # keep system, drop user+assistant
        else:
            messages = messages[2:]  # drop user+assistant
        if not messages:
            break
        entry = copy.deepcopy(entry)
        entry[prompt_key] = messages
        n = _length(entry)
        if n <= max_tokens:
            return (n, entry)

    return None


# Module-level ref for worker processes (set by _init_worker)
_worker_processor = None


def _init_worker(model_name: str):
    """Load processor in each worker process. VIDEO_MAX_FRAMES env should be set before import."""
    global _worker_processor
    _worker_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def _process_one(args: tuple) -> tuple:
    """Process one entry. Returns (index, token_count_or_none, entry). None = error."""
    i, entry, data_source_dir, max_video_frames, max_tokens, truncate_overlong = args
    try:
        n = compute_prompt_length(
            entry,
            _worker_processor,
            data_source_dir=data_source_dir,
            max_video_frames=max_video_frames,
        )
        if n is not None and n > max_tokens and truncate_overlong:
            result = try_truncate_entry(
                entry, _worker_processor, max_tokens, data_source_dir, max_video_frames
            )
            if result is not None:
                n, entry = result
        return (i, n, entry)
    except Exception:
        return (i, None, entry)


def main():
    parser = argparse.ArgumentParser(description="Filter dataset by prompt token length")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Input JSONL files. Default: data/combined_train_demo_only.json data/combined_valid_demo_only.json",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for resolving relative paths. Default: dirname of first input",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max prompt tokens (default: 4096)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name for processor",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Output suffix, e.g. filtered_4096. Default: filtered_<max_tokens>",
    )
    parser.add_argument(
        "--only-check-hb",
        action="store_true",
        help="Only run token-length check on HB entries. CLIMB entries pass through unchanged.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8). Reduce if OOM / BrokenProcessPool.",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=None,
        help="Hard cap on video frames per video (default: 4). Overrides VIDEO_MAX_FRAMES env.",
    )
    parser.add_argument(
        "--truncate-overlong",
        action="store_true",
        help="Truncate overlong samples (shorten system, drop turns) instead of skipping them.",
    )
    parser.add_argument(
        "--subsample-pct",
        type=float,
        default=None,
        help="Take this %% from each data_source (e.g. 10 for 10%%). Stratified subsample before filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    args = parser.parse_args()

    if args.max_video_frames is None:
        args.max_video_frames = int(os.environ.get("VIDEO_MAX_FRAMES", "4"))

    if args.inputs is None:
        data_dir = Path(PROJECT_ROOT) / "data"
        args.inputs = [
            str(data_dir / "combined_train_demo_only.json"),
            str(data_dir / "combined_valid_demo_only.json"),
        ]

    suffix = args.suffix or f"filtered_{args.max_tokens}"
    data_dir = args.data_dir or str(Path(args.inputs[0]).resolve().parent)

    def is_hb_entry(ent):
        if not args.only_check_hb:
            return True
        try:
            ei = ent.get("extra_info")
            if isinstance(ei, str):
                ei = json.loads(ei)
            return (ei or {}).get("source_dataset") == "human_behaviour"
        except Exception:
            return True

    for input_path in args.inputs:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"  Skip (not found): {input_path}", file=sys.stderr)
            continue

        stem = input_path.stem
        out_path = input_path.parent / f"{stem}_{suffix}.json"
        if out_path == input_path:
            out_path = input_path.parent / f"{stem}_{suffix}.json"

        entries = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        # Subsample 10% (or args.subsample_pct) per data_source
        if args.subsample_pct is not None:
            original_count = len(entries)
            rng = random.Random(args.seed)
            by_source = defaultdict(list)
            for i, e in enumerate(entries):
                by_source[e.get("data_source", "unknown")].append((i, e))
            subsampled = []
            for src, group in by_source.items():
                k = max(1, int(len(group) * args.subsample_pct / 100))
                subsampled.extend(rng.sample(group, min(k, len(group))))
            subsampled.sort(key=lambda x: x[0])
            entries = [e for _, e in subsampled]
            print(f"  Subsampled {original_count:,} -> {len(entries):,} ({args.subsample_pct}%% per category, {len(by_source)} sources)", file=sys.stderr)

        # Entries that pass through without token check (CLIMB when only-check-hb)
        pass_through = []
        to_compute = []  # (index, entry, data_dir, max_video_frames, max_tokens, truncate_overlong)
        for i, entry in enumerate(entries):
            if args.only_check_hb and not is_hb_entry(entry):
                pass_through.append((i, entry))
            else:
                to_compute.append((i, entry, data_dir, args.max_video_frames, args.max_tokens, args.truncate_overlong))

        total = len(to_compute)
        print(f"Processing {input_path}: {len(entries):,} entries" + (" (HB-only check)" if args.only_check_hb else ""), file=sys.stderr)
        if args.workers > 1:
            print(f"  Using {args.workers} workers", file=sys.stderr)
            results = []
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.model,)) as pool:
                futures = {pool.submit(_process_one, t): t[0] for t in to_compute}
                for future in as_completed(futures):
                    results.append(future.result())
                    done += 1
                    if done % 500 == 0 or done == total:
                        print(f"  ... {done:,}/{total:,}", file=sys.stderr)
            results.sort(key=lambda r: r[0])
        else:
            print(f"Loading processor from {args.model}...", file=sys.stderr)
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
            results = []
            for j, (i, entry, _, max_vf, max_tok, trunc) in enumerate(to_compute):
                try:
                    n = compute_prompt_length(
                        entry, processor, data_source_dir=data_dir, max_video_frames=max_vf
                    )
                    if n is not None and n > max_tok and trunc:
                        result = try_truncate_entry(
                            entry, processor, max_tok, data_dir, max_vf
                        )
                        if result is not None:
                            n, entry = result
                    results.append((i, n, entry))
                except Exception as e:
                    results.append((i, None, entry))
                    if len([r for r in results if r[1] is None]) <= 3:
                        print(f"  Error at index {i}: {e}", file=sys.stderr)
                if (j + 1) % 500 == 0 or (j + 1) == total:
                    print(f"  ... {j + 1:,}/{total:,}", file=sys.stderr)

        # Merge: pass_through always kept; to_compute: keep if n <= max_tokens
        index_to_entry = {i: entry for i, entry in pass_through}
        for i, n, entry in results:
            if n is not None and n <= args.max_tokens:
                index_to_entry[i] = entry
        kept = [index_to_entry[i] for i in sorted(index_to_entry.keys())]
        skipped = sum(1 for _, n, _ in results if n is not None and n > args.max_tokens)
        errors = sum(1 for _, n, _ in results if n is None)

        with open(out_path, "w") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        climb_str = f", CLIMB passed {len(pass_through)}" if args.only_check_hb and pass_through else ""
        print(
            f"  -> {out_path}: kept {len(kept):,}, skipped {skipped}, errors {errors}{climb_str}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
