"""Export every eligible row of a split half to a task JSONL for generation.

Runs on the cluster (where the parquets live); generation also runs cluster-side
(compute nodes reach the teacher endpoint). Every SFT row gets a teacher trace,
so there is no sampling here -- the split is the sampling. Each task carries the FULL original prompt (all system+user turns
flattened), every media reference in original order, its ground truth (for
laptop-side validation only -- the generator strips it before building
requests), and answer_style: sources in schema.OPEN_SOURCES are free text with
no exact-match gate, everything else is gradable.

    python mirl_ext/sft/scripts/export_sft_tasks.py --out .../sft_tasks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import (  # noqa: E402
    DATA_ROOT,
    FAMILIES,
    OPEN_SOURCES,
    media_refs,
    prompt_text,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--out", default=f"{DATA_ROOT}/split/sft_tasks.jsonl")
    ap.add_argument(
        "--half",
        default="sft",
        choices=["sft", "rl"],
        help="split half to export ('rl' only builds the episode-generator support pool)",
    )
    ap.add_argument(
        "--families", nargs="*", default=None, help="subset of families (default: all)"
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    half_root = Path(args.split_root) / args.half
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = args.families or FAMILIES
    total = 0
    with out_path.open("w") as fh:
        for family in wanted:
            src = half_root / f"{family}.parquet"
            if not src.exists():
                print(f"[skip] {src} not found")
                continue
            rows = pq.read_table(src).to_pylist()
            # Keep original parquet positions: uid must join back to the row.
            eligible = [
                (i, row)
                for i, row in enumerate(rows)
                if str((row.get("reward_model") or {}).get("ground_truth") or "").strip()
            ]
            exported_sources: collections.Counter = collections.Counter()
            labels = set()
            n_media = n_open = 0
            for i, row in eligible:
                data_source = row.get("data_source")
                gt = (row.get("reward_model") or {}).get("ground_truth")
                images, video_path = media_refs(row)
                style = "open" if str(data_source) in OPEN_SOURCES else "closed"
                task = {
                    # build_sft_parquet joins completions back on uid.
                    "uid": f"{family}#{i}",
                    "family": family,
                    "row_index": i,
                    "data_source": data_source,
                    "prompt": prompt_text(row),
                    "ground_truth": gt,
                    "image_paths": images,
                    "video_path": video_path,
                    "answer_style": style,
                }
                fh.write(json.dumps(task) + "\n")
                exported_sources[str(data_source)] += 1
                labels.add(gt)
                n_media += bool(images or video_path)
                n_open += style == "open"
                total += 1
            print(
                f"{family:22s} half_rows={len(rows):6d} exported={len(eligible):6d} "
                f"labels={len(labels):5d} with_media={n_media} open={n_open}"
            )

    print(f"\nwrote {total} tasks -> {out_path}")
    print("Copy to the machine with internet access, stage media, then run gen_sft_targets.py.")


if __name__ == "__main__":
    main()
