"""Freeze the exact staged video frames seen by a set of accepted traces.

Shared staging directories can accumulate a different frame count after later
pipeline runs. This tool reads the archived staged task JSONL instead of
glob-discovering frames, validates every accepted trace/task join, and creates
a versioned hard-link tree suitable for ``build_sft_parquet.py
--frames-from-staging``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from mirl_ext.data.schema import iter_jsonl, media_stem
from mirl_ext.sft.scripts.build_sft_parquet import accepted_traces


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=Path, required=True)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="maximum staged frames per video; naturally short clips may have fewer",
    )
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse existing output: {args.out_root}")

    traces = accepted_traces(args.traces)
    found: set[str] = set()
    video_tasks = unique_videos = linked_frames = 0
    seen_videos: set[tuple[str, str]] = set()
    frame_counts: Counter[int] = Counter()

    for task_path in args.tasks:
        for task in iter_jsonl(task_path):
            uid = task["uid"]
            trace = traces.get(uid)
            if trace is None:
                continue
            if uid in found:
                raise ValueError(f"duplicate accepted task uid: {uid}")
            if task.get("ground_truth") != trace.get("ground_truth"):
                raise ValueError(f"{uid}: task/trace ground-truth mismatch")
            found.add(uid)

            video_path = task.get("video_path")
            if not video_path:
                continue
            frames = task.get("frame_paths") or []
            if not 0 < len(frames) <= args.max_frames:
                raise ValueError(f"{uid}: expected 1..{args.max_frames} archived frames, got {len(frames)}")
            video_tasks += 1
            frame_counts[len(frames)] += 1
            video_key = (task["family"], video_path)
            if video_key in seen_videos:
                continue
            seen_videos.add(video_key)
            unique_videos += 1

            stem = media_stem(video_path)
            destination = args.out_root / task["family"]
            destination.mkdir(parents=True, exist_ok=True)
            for raw_frame in frames:
                source = Path(raw_frame)
                if not source.is_file():
                    raise FileNotFoundError(f"{uid}: missing archived frame: {source}")
                if not source.name.startswith(f"{stem}_f"):
                    raise ValueError(f"{uid}: frame stem does not match video path: {source.name}")
                target = destination / source.name
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                linked_frames += 1

    missing = set(traces) - found
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        raise ValueError(f"{len(missing)} accepted traces have no archived task; first: {preview}")

    manifest = {
        "max_frames_per_video": args.max_frames,
        "video_task_frame_count_histogram": dict(sorted(frame_counts.items())),
        "accepted_tasks": len(found),
        "video_tasks": video_tasks,
        "unique_videos": unique_videos,
        "linked_frames": linked_frames,
        "task_files": {str(path): sha256(path) for path in args.tasks},
        "trace_files": {str(path): sha256(path) for path in args.traces},
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
