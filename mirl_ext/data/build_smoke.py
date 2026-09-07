#!/usr/bin/env python3
"""Build a tiny, deterministic mixed-modality parquet for Qwen3.5 smoke tests."""

from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mirl_ext.data.schema import (
    HUMAN_BEHAVIOUR_SOURCES,
    MEDICAL_SOURCES,
    OPEN_SOURCES,
    TACTILE_SOURCES,
)

SELECTIONS = {
    "ecg_train.parquet": 2,
    "climb_train.parquet": 2,
    "human_behaviour_train_closed.parquet": 2,
    "tactile_train_closed.parquet": 2,
}
ACTIVE_SMOKE_SOURCES = frozenset({"ecg"} | MEDICAL_SOURCES | HUMAN_BEHAVIOUR_SOURCES | TACTILE_SOURCES) - OPEN_SOURCES


def _prompt_chars(row: dict) -> int:
    return sum(len(message.get("content", "")) for message in row["prompt"])


def _media_paths(row: dict) -> list[str]:
    paths = [image["image"] for image in row.get("images") or []]
    paths.extend(video["video"] for video in row.get("videos") or [])
    return paths


def _usable(row: dict, max_video_bytes: int) -> bool:
    paths = _media_paths(row)
    if not paths or not all(os.path.isfile(path) for path in paths):
        return False
    return all(
        not path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")) or os.path.getsize(path) <= max_video_bytes
        for path in paths
    )


def _closed(row: dict) -> bool:
    return str(row.get("data_source", "")) in ACTIVE_SMOKE_SOURCES


def build(data_root: Path, output: Path, max_video_bytes: int, scan_rows: int = 2048) -> None:
    selected_tables = []
    for filename, count in SELECTIONS.items():
        parquet = pq.ParquetFile(data_root / filename)
        candidates: list[tuple[int, int, dict]] = []
        row_index = 0
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if row_index >= scan_rows:
                    break
                if _closed(row) and _usable(row, max_video_bytes):
                    candidates.append((_prompt_chars(row), row_index, row))
                row_index += 1
            if row_index >= scan_rows:
                break
        candidates = heapq.nsmallest(count, candidates, key=lambda candidate: candidate[:2])
        if len(candidates) < count:
            raise RuntimeError(f"{filename}: found {len(candidates)} closed usable rows, need {count}")
        selected_tables.append(
            pa.Table.from_pylist([candidate[2] for candidate in candidates], schema=parquet.schema_arrow)
        )

    combined = pa.concat_tables(selected_tables, promote_options="default")
    # Each source parquet carries HF feature metadata for its original schema.
    # After Arrow promotes source-specific null fields, those stale features no
    # longer describe the merged table; let HF infer them from the unified schema.
    combined = combined.replace_schema_metadata(None)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output, compression="zstd")

    sources = combined.column("data_source").to_pylist()
    print(f"wrote {combined.num_rows} rows to {output}")
    print("data_sources=" + ",".join(sources))
    for row in combined.to_pylist():
        print(f"{row['data_source']}: {_media_paths(row)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-video-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--scan-rows", type=int, default=2048)
    args = parser.parse_args()
    build(args.data_root, args.output, args.max_video_bytes, args.scan_rows)


if __name__ == "__main__":
    main()
