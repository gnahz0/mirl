"""Convert and combine HB + CLIMB + Tactile datasets into the unified annotation format.

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

Run: ``python scripts/combine_datasets.py`` (defaults use the path constants below
for train demo). For val/test demo, override ``--hb`` / ``--hb-unified``, ``--climb``,
``--tactile``, and ``--output``.

Pre-filtered HB (unified JSONL from ``filter_by_token_limit --hb-only``)::

  python scripts/combine_datasets.py \\
    --hb-unified data/hb_only_filtered_8192_checkpoint_combined_train_demo_only.json \\
    --output data/combined_train_demo_only_filtered_8192.json

To drop overlong Human Behaviour rows only (keep CLIMB/tactile unchanged) and train
with ``data.filter_overlong_prompts=False``, run on the **combined** JSON lines::

  python scripts/filter_by_token_limit.py --only-check-hb --max-tokens 8192 --max-video-frames 4

HB-only filtered JSONL (no combined ``*_filtered_*`` files): add ``--hb-only`` (writes
``data/hb_only_filtered_8192.json``). The filter defaults to 2 workers; use ``--workers 1``
if you hit OOM / BrokenProcessPool. It falls back to sequential if the process pool dies.
"""

import argparse
import json
import os
import sys

# --- File paths (this node) ---
HB_MEDIA_ROOT = "/scratch/keane/human_behaviour/human_behaviour_data"
HB_JSONL_TRAIN = f"{HB_MEDIA_ROOT}/v5_train_upd.jsonl"
HB_JSONL_TEST = f"{HB_MEDIA_ROOT}/v5_test_upd.jsonl"
CLIMB_JSONL_TRAIN_DEMO = "/orcd/compute/ppliang/001/high_modality/geom_train_demo_only.jsonl"
CLIMB_JSONL_VAL_DEMO = "/orcd/compute/ppliang/001/high_modality/geom_valid_demo_only.jsonl"
CLIMB_MEDIA_ROOT = "/orcd/compute/ppliang/001/high_modality"
TACTILE_JSON_TRAIN = "/orcd/compute/ppliang/001/raofu/3DHaptic/annotation_verl_split_date_train.json"
TACTILE_JSON_TEST = "/orcd/compute/ppliang/001/raofu/3DHaptic/annotation_verl_split_date_test.json"
TACTILE_MEDIA_ROOT = "/orcd/compute/ppliang/001/raofu/3DHaptic"
OUTPUT_COMBINED_TRAIN_DEMO = "/home/alecz/mirl/data/combined_train_demo_only_filtered_8192.json"
OUTPUT_COMBINED_VAL_DEMO = "/home/alecz/mirl/data/combined_valid_demo_only_filtered_8192.json"

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


def convert_tactile_entry(entry: dict, idx: int, root: str) -> dict:
    """Pass through a tactile entry, resolving relative video paths."""
    out = {
        "data_source": entry["data_source"],
        "prompt": entry["prompt"],
        "images": [],
        "videos": resolve_video_paths(entry.get("videos", []), root),
        "audios": [],
        "reward_model": entry["reward_model"],
        "extra_info": json.dumps(entry["extra_info"]) if isinstance(entry.get("extra_info"), dict) else entry.get("extra_info", "{}"),
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


def interleave(sources: list[list[dict]]) -> list[dict]:
    """Round-robin interleave multiple lists proportionally."""
    total = sum(len(s) for s in sources)
    if total == 0:
        return []
    indices = [0] * len(sources)
    lengths = [len(s) for s in sources]
    result = []
    while len(result) < total:
        for i, src in enumerate(sources):
            if indices[i] < lengths[i]:
                frac_emitted = indices[i] / lengths[i] if lengths[i] else 1.0
                frac_overall = len(result) / total
                if frac_emitted <= frac_overall or all(
                    indices[j] >= lengths[j] for j in range(len(sources)) if j != i
                ):
                    result.append(src[indices[i]])
                    indices[i] += 1
                    break
    return result


def read_unified_jsonl(path: str) -> list[dict]:
    """Load JSONL where each line is already in unified verl format (no conversion)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Combine HB + CLIMB + Tactile into unified format")
    parser.add_argument(
        "--hb-unified",
        default=None,
        help=(
            "HB JSONL already in unified format (e.g. data/hb_only_filtered_8192_checkpoint_*.json from "
            "filter_by_token_limit --hb-only). If set, used instead of raw --hb + convert_hb_entry."
        ),
    )
    parser.add_argument("--hb", default=HB_JSONL_TRAIN, help="Path to human behaviour JSONL (raw v5 schema)")
    parser.add_argument("--hb-root", default=HB_MEDIA_ROOT, help="Base directory for HB media paths")
    parser.add_argument("--climb", default=CLIMB_JSONL_TRAIN_DEMO, help="Path to CLIMB/medical JSONL")
    parser.add_argument("--climb-root", default=CLIMB_MEDIA_ROOT, help="Base directory for CLIMB media paths")
    parser.add_argument("--tactile", default=TACTILE_JSON_TRAIN, help="Path to tactile JSON")
    parser.add_argument("--tactile-root", default=TACTILE_MEDIA_ROOT, help="Base directory for tactile video paths")
    parser.add_argument(
        "--output",
        default=OUTPUT_COMBINED_TRAIN_DEMO,
        help="Output JSON path (newline-delimited JSON objects)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    all_sources = []

    if args.hb_unified and os.path.isfile(args.hb_unified):
        print(f"Reading HB (unified / pre-filtered): {args.hb_unified}", file=sys.stderr)
        hb_entries = read_unified_jsonl(args.hb_unified)
        print(f"  => {len(hb_entries):,} HB entries", file=sys.stderr)
        all_sources.append(hb_entries)
    elif args.hb and os.path.isfile(args.hb):
        print(f"Reading HB: {args.hb}", file=sys.stderr)
        hb_entries = read_all(args.hb, convert_hb_entry, args.hb_root)
        print(f"  => {len(hb_entries):,} HB entries", file=sys.stderr)
        all_sources.append(hb_entries)
    else:
        print(f"Skipping HB (use --hb-unified or raw --hb; not found: {args.hb})", file=sys.stderr)

    if args.climb and os.path.isfile(args.climb):
        print(f"Reading CLIMB: {args.climb}", file=sys.stderr)
        climb_entries = read_all(args.climb, convert_climb_entry, args.climb_root)
        print(f"  => {len(climb_entries):,} CLIMB entries", file=sys.stderr)
        all_sources.append(climb_entries)
    else:
        print(f"Skipping CLIMB (not found: {args.climb})", file=sys.stderr)

    if args.tactile and os.path.isfile(args.tactile):
        print(f"Reading Tactile: {args.tactile}", file=sys.stderr)
        tactile_entries = read_all(args.tactile, convert_tactile_entry, args.tactile_root)
        print(f"  => {len(tactile_entries):,} Tactile entries", file=sys.stderr)
        all_sources.append(tactile_entries)
    else:
        print(f"Skipping Tactile (not found: {args.tactile})", file=sys.stderr)

    if not all_sources:
        print("ERROR: no source datasets found, nothing to write.", file=sys.stderr)
        sys.exit(1)

    # Interleave so PyArrow sees all column types (list<string> for images,
    # audios, etc.) in its first inference batch.
    print("Interleaving and writing...", file=sys.stderr)
    merged = interleave(all_sources)
    total = len(merged)
    with open(args.output, "w") as out:
        for i, entry in enumerate(merged):
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if (i + 1) % 200_000 == 0:
                print(f"  ... written {i + 1:,}/{total:,}", file=sys.stderr)

    print(f"Total: {total:,} entries -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
