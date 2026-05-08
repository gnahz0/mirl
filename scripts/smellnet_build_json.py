"""
Build training/test JSONL for SmellNet subplot images.

Outputs newline-delimited JSON files with one entry per sample:
  {"image": "/abs/path/to/img.png", "label": "cinnamon", "split": "train", "task": "base"}
  {"image": "/abs/path/to/img.png", "label": "Almond20_Orange80", "split": "train", "task": "mixture"}

Usage:
    python scripts/smellnet_build_json.py \
        --image_dir /home/alecz/scratch/alecz/SmellNet_subplot \
        --output_dir /home/alecz/scratch/alecz/SmellNet_subplot
"""

import argparse
import json
import os


def collect_base_data(image_dir: str) -> list[dict]:
    """base_data/{training,testing}/<substance>/<substance>_N.png"""
    entries = []
    for split_name, split_label in [("training", "train"), ("testing", "test")]:
        split_dir = os.path.join(image_dir, "base_data", split_name)
        if not os.path.isdir(split_dir):
            continue
        for substance in sorted(os.listdir(split_dir)):
            subst_dir = os.path.join(split_dir, substance)
            if not os.path.isdir(subst_dir):
                continue
            for fname in sorted(os.listdir(subst_dir)):
                if not fname.endswith(".png"):
                    continue
                entries.append({
                    "image": os.path.join(subst_dir, fname),
                    "label": substance,
                    "split": split_label,
                    "task": "base",
                })
    return entries


def collect_mixture_data(image_dir: str) -> list[dict]:
    """mixture_data/{training_seen,training_new,test_seen,test_unseen}/<mixture_file>.png"""
    split_map = {
        "training_seen": "train",
        "training_new": "train",
        "test_seen": "test",
        "test_unseen": "test",
    }
    entries = []
    for split_dir_name, split_label in split_map.items():
        split_dir = os.path.join(image_dir, "mixture_data", split_dir_name)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".png"):
                continue
            mixture_name = fname.split(".")[0]
            entries.append({
                "image": os.path.join(split_dir, fname),
                "label": mixture_name,
                "split": split_label,
                "task": "mixture",
                "mixture_split": split_dir_name,
            })
    return entries


def write_jsonl(entries: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root of subplot image directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write JSONL files")
    args = parser.parse_args()

    base = collect_base_data(args.image_dir)
    mixture = collect_mixture_data(args.image_dir)

    base_train = [e for e in base if e["split"] == "train"]
    base_test = [e for e in base if e["split"] == "test"]
    mix_train = [e for e in mixture if e["split"] == "train"]
    mix_test = [e for e in mixture if e["split"] == "test"]

    print(f"Base:    {len(base_train)} train, {len(base_test)} test ({len(set(e['label'] for e in base))} classes)")
    print(f"Mixture: {len(mix_train)} train, {len(mix_test)} test ({len(set(e['label'] for e in mixture))} classes)")

    write_jsonl(base_train, os.path.join(args.output_dir, "base_train.jsonl"))
    write_jsonl(base_test, os.path.join(args.output_dir, "base_test.jsonl"))
    write_jsonl(mix_train, os.path.join(args.output_dir, "mixture_train.jsonl"))
    write_jsonl(mix_test, os.path.join(args.output_dir, "mixture_test.jsonl"))

    all_train = base_train + mix_train
    all_test = base_test + mix_test
    write_jsonl(all_train, os.path.join(args.output_dir, "all_train.jsonl"))
    write_jsonl(all_test, os.path.join(args.output_dir, "all_test.jsonl"))

    print(f"\nTotal: {len(all_train)} train, {len(all_test)} test")


if __name__ == "__main__":
    main()
