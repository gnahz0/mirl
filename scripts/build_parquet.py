#!/usr/bin/env python
"""Convert the per-dataset JSONL splits in data/ to parquet with one unified schema.

The six datasets differ in which media columns are populated: the time-series
datasets (smellnet/ecg/haptic_ts) have empty `videos`, while human_behaviour/
tactile have empty `images`. HF's JSON loader infers empty lists as List(null),
which won't align with the populated List(struct) in sibling files, so
`datasets.concatenate_datasets` (used by RLHFDataset) fails on multi-file loads.

Casting every file to one explicit schema and writing parquet fixes this:
parquet stores the schema in its metadata, so the loader never re-infers and
all six files concatenate cleanly.

Usage: python scripts/build_parquet.py [--data-dir data]
"""
import argparse
import os

import datasets
from datasets import Features, Value

NAMES = ["smellnet", "ecg", "haptic_ts", "climb", "human_behaviour", "tactile"]

UNIFIED = Features(
    {
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": [{"image": Value("string")}],
        "videos": [
            {
                "video": Value("string"),
                "min_frames": Value("int64"),
                "max_frames": Value("int64"),
            }
        ],
        "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
        "extra_info": Value("string"),
    }
)


def _normalize(record):
    record["videos"] = [
        {
            "video": v.get("video"),
            "min_frames": v.get("min_frames"),
            "max_frames": v.get("max_frames"),
        }
        for v in (record.get("videos") or [])
        if isinstance(v, dict)
    ]
    record["images"] = [
        {"image": im.get("image")}
        for im in (record.get("images") or [])
        if isinstance(im, dict)
    ]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    datasets.disable_progress_bars()

    for split in ["train", "valid"]:
        for name in NAMES:
            src = os.path.join(args.data_dir, f"{name}_{split}.json")
            if not os.path.exists(src):
                print(f"  SKIP missing {src}")
                continue
            ds = datasets.load_dataset("json", data_files=src)["train"]
            ds = ds.map(_normalize)
            ds = ds.cast(UNIFIED)
            out = os.path.join(args.data_dir, f"{name}_{split}.parquet")
            ds.to_parquet(out)
            print(f"  {name:16} {split:5} rows={len(ds):7d} -> {out}")
    print("done")


if __name__ == "__main__":
    main()
