"""Convert and combine HB + CLIMB datasets into the unified annotation format.

Output format per entry:
{
  "data_source": "<reward dispatch key>",
  "prompt": [
    {"role": "system", "content": "<domain system prompt>"},
    {"role": "user",   "content": "<media tag>\\n<question text>"}
  ],
  "images": [...],          # climb only  (absolute paths)
  "videos": [...],          # if present   (absolute paths)
  "audios": [...],          # HB only     (absolute paths)
  "reward_model": {"style": "rule", "ground_truth": "<answer>"},
  "extra_info": { ... }
}

Usage:
  python scripts/combine_datasets.py \
      --hb /scratch/keane/human_behaviour_data/v5_test_upd.jsonl \
      --hb-root /scratch/keane/human_behaviour_data \
      --climb /home/alecz/scratch/high_modality/geom_valid_demo_only.jsonl \
      --climb-root /home/alecz/scratch/high_modality \
      --output combined_valid_demo_only.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

HB_SYSTEM_PROMPT = (
    "You are an expert in analyzing human behaviour from multimodal signals "
    "(speech, video, text). Examine the provided input carefully and answer the "
    "question. You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final answer MUST BE wrapped in \\boxed{}."
)

CLIMB_SYSTEM_PROMPT = (
    "You are an expert in medical image analysis. Examine the provided medical image "
    "carefully and answer the clinical question. You FIRST think about the reasoning "
    "process as an internal monologue and then provide the final answer. The reasoning "
    "process MUST BE enclosed within <think> </think> tags. The final answer MUST BE "
    "wrapped in \\boxed{}."
)


def resolve_paths(paths: list[str], root: str) -> list[str]:
    """Make relative paths absolute by joining with root dir."""
    resolved = []
    for p in paths:
        if os.path.isabs(p):
            resolved.append(p)
        else:
            resolved.append(os.path.join(root, p))
    return resolved


def resolve_image_paths(paths: list, root: str) -> list[dict]:
    """Convert image paths to dicts with an 'image' key (matching tactile format)."""
    result = []
    for p in paths:
        if isinstance(p, dict):
            if "image" in p and not os.path.isabs(p["image"]):
                p["image"] = os.path.join(root, p["image"])
            result.append(p)
        else:
            abs_path = p if os.path.isabs(p) else os.path.join(root, p)
            result.append({"image": abs_path})
    return result


def resolve_video_paths(paths: list, root: str, max_frames: int = 8) -> list[dict]:
    """Convert video paths to dicts with a 'video' key (matching tactile format)."""
    result = []
    for p in paths:
        if isinstance(p, dict):
            if "video" in p and not os.path.isabs(p["video"]):
                p["video"] = os.path.join(root, p["video"])
            p.setdefault("max_frames", max_frames)
            result.append(p)
        else:
            abs_path = p if os.path.isabs(p) else os.path.join(root, p)
            result.append({"video": abs_path, "max_frames": max_frames})
    return result


def resolve_audio_paths(paths: list, root: str) -> list[dict]:
    """Convert audio paths to dicts with an 'audio' key (matching tactile format)."""
    result = []
    for p in paths:
        if isinstance(p, dict):
            if "audio" in p and not os.path.isabs(p["audio"]):
                p["audio"] = os.path.join(root, p["audio"])
            result.append(p)
        else:
            abs_path = p if os.path.isabs(p) else os.path.join(root, p)
            result.append({"audio": abs_path})
    return result


def convert_hb_entry(entry: dict, idx: int, root: str) -> dict:
    """Convert a human-behaviour JSONL entry to the unified format."""
    dataset = entry.get("dataset", "unknown")

    prompt = [
        {"role": "system", "content": HB_SYSTEM_PROMPT},
        {"role": "user", "content": entry["problem"]},
    ]

    extra = {
        "index": idx,
        "source_dataset": "human_behaviour",
        "dataset": dataset,
        "modality_signature": entry.get("modality_signature", ""),
    }
    if entry.get("texts"):
        extra["texts"] = entry["texts"]
    if entry.get("ext_video_feats"):
        extra["ext_video_feats"] = resolve_paths(entry["ext_video_feats"], root)
    if entry.get("ext_audio_feats"):
        extra["ext_audio_feats"] = resolve_paths(entry["ext_audio_feats"], root)

    out = {
        "data_source": dataset,
        "prompt": prompt,
        "images": resolve_image_paths(entry["images"], root) if entry.get("images") else [],
        "videos": resolve_video_paths(entry["videos"], root) if entry.get("videos") else [],
        "audios": resolve_audio_paths(entry["audios"], root) if entry.get("audios") else [],
        "reward_model": {"style": "rule", "ground_truth": entry["answer"]},
        "extra_info": json.dumps(extra),
    }

    return out


def convert_climb_entry(entry: dict, idx: int, root: str) -> dict:
    """Convert a CLIMB/medical JSONL entry to the unified format."""
    data_source = entry.get("data_source", "unknown")

    prompt = [
        {"role": "system", "content": CLIMB_SYSTEM_PROMPT},
        {"role": "user", "content": entry["problem"]},
    ]

    extra = {
        "index": idx,
        "source_dataset": "climb",
        "dataset": entry.get("dataset", ""),
        "data_source": data_source,
    }
    if entry.get("segmentation_path"):
        extra["segmentation_path"] = os.path.join(root, entry["segmentation_path"]) if not os.path.isabs(entry["segmentation_path"]) else entry["segmentation_path"]
    if entry.get("bbox"):
        extra["bbox"] = entry["bbox"]

    out = {
        "data_source": data_source,
        "prompt": prompt,
        "images": resolve_image_paths(entry["images"], root) if entry.get("images") else [],
        "videos": resolve_video_paths(entry["videos"], root) if entry.get("videos") else [],
        "audios": [],
        "reward_model": {"style": "rule", "ground_truth": entry["answer"]},
        "extra_info": json.dumps(extra),
    }

    return out


def read_all(path: str, converter, root: str) -> list[dict]:
    """Read and convert all entries from a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(converter(json.loads(line), len(entries), root))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Combine HB + CLIMB into unified format")
    parser.add_argument("--hb", required=True, help="Path to human behaviour JSONL")
    parser.add_argument("--hb-root", required=True, help="Base directory for HB media paths")
    parser.add_argument("--climb", required=True, help="Path to CLIMB/medical JSONL")
    parser.add_argument("--climb-root", required=True, help="Base directory for CLIMB media paths")
    parser.add_argument("--output", required=True, help="Output .json path")
    args = parser.parse_args()

    print(f"Reading HB: {args.hb}", file=sys.stderr)
    hb_entries = read_all(args.hb, convert_hb_entry, args.hb_root)
    print(f"  => {len(hb_entries):,} HB entries", file=sys.stderr)

    print(f"Reading CLIMB: {args.climb}", file=sys.stderr)
    climb_entries = read_all(args.climb, convert_climb_entry, args.climb_root)
    print(f"  => {len(climb_entries):,} CLIMB entries", file=sys.stderr)

    # Interleave so PyArrow sees both column types (list<string> for images
    # and audios) in its first inference batch. Without this, empty lists []
    # get typed as list<null> and later batches with actual paths fail to cast.
    print("Interleaving and writing...", file=sys.stderr)
    n_hb, n_climb = len(hb_entries), len(climb_entries)
    total = n_hb + n_climb
    with open(args.output, "w") as out:
        hi, ci, written = 0, 0, 0
        # Ratio-based interleave: emit entries at their natural frequency
        while hi < n_hb or ci < n_climb:
            if hi < n_hb and (ci >= n_climb or hi / n_hb <= ci / n_climb):
                out.write(json.dumps(hb_entries[hi], ensure_ascii=False) + "\n")
                hi += 1
            else:
                out.write(json.dumps(climb_entries[ci], ensure_ascii=False) + "\n")
                ci += 1
            written += 1
            if written % 200_000 == 0:
                print(f"  ... written {written:,}/{total:,}", file=sys.stderr)

    print(f"Total: {total:,} entries -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
