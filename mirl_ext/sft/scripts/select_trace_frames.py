"""Freeze the exact staged media seen by a set of accepted traces.

Shared staging directories can accumulate a different frame count after later
pipeline runs. This tool reads the archived staged task JSONL instead of
glob-discovering frames, validates every accepted trace/task join, and creates
a versioned hard-link tree suitable for ``build_sft_parquet.py
--media-from-staging``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import iter_jsonl, media_stem  # noqa: E402
from mirl_ext.sft.artifacts import (  # noqa: E402
    is_sha256_digest,
    sha256,
    task_fingerprint,
    verify_media_hashes,
)
from mirl_ext.sft.traces import accepted_traces  # noqa: E402

MAX_FROZEN_VIDEO_FRAMES = 24


def _validate_task_trace(task: dict, trace: dict) -> None:
    """Require an archived task and paid trace to describe the same input."""
    uid = task["uid"]
    if not is_sha256_digest(task.get("source_row_fingerprint")):
        raise ValueError(f"{uid}: task has missing or invalid source_row_fingerprint")
    for field in ("family", "data_source", "ground_truth", "source_row_fingerprint"):
        if task.get(field) != trace.get(field):
            raise ValueError(
                f"{uid}: task/trace {field} mismatch: "
                f"{task.get(field)!r} != {trace.get(field)!r}"
            )
    task_version = task.get("staging_version")
    trace_version = trace.get("staging_version")
    if not task_version or not trace_version:
        raise ValueError(f"{uid}: task and trace both need staging_version provenance")
    if task_version != trace_version:
        raise ValueError(
            f"{uid}: task/trace staging_version mismatch: "
            f"{task_version!r} != {trace_version!r}"
        )
    fingerprint = task.get("task_fingerprint")
    if not fingerprint or fingerprint != task_fingerprint(task):
        raise ValueError(f"{uid}: task has missing or invalid task_fingerprint")
    if trace.get("task_fingerprint") != fingerprint:
        raise ValueError(f"{uid}: task/trace task_fingerprint mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=Path, required=True)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FROZEN_VIDEO_FRAMES,
        help="integrity ceiling across modalities (tactile <=24; generic RGB =8)",
    )
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse existing output: {args.out_root}")

    traces = accepted_traces(args.traces)
    found: set[str] = set()
    image_tasks = unique_images = linked_images = 0
    video_tasks = unique_videos = linked_frames = 0
    seen_images: dict[tuple[str, str], str] = {}
    seen_videos: dict[tuple[str, str], tuple[str, ...]] = {}
    frame_counts: Counter[int] = Counter()
    media_digest_cache: dict[str, str] = {}

    for task_path in args.tasks:
        for task in iter_jsonl(task_path):
            uid = task["uid"]
            trace = traces.get(uid)
            if trace is None:
                continue
            if uid in found:
                raise ValueError(f"duplicate accepted task uid: {uid}")
            _validate_task_trace(task, trace)
            teacher_media = task.get("frame_paths") or task.get("image_paths") or []
            verify_media_hashes(
                map(Path, teacher_media),
                task.get("media_sha256") or [],
                media_digest_cache,
            )
            found.add(uid)

            video_path = task.get("video_path")
            images = task.get("image_paths") or []
            if bool(video_path) == bool(images):
                raise ValueError(f"{uid}: expected exactly one of staged images/video")
            destination = args.out_root / task["family"]
            destination.mkdir(parents=True, exist_ok=True)
            if not video_path:
                image_tasks += 1
                for raw_image, digest in zip(
                    images, task["media_sha256"], strict=True
                ):
                    source = Path(raw_image)
                    image_key = (task["family"], source.name)
                    if image_key in seen_images:
                        if seen_images[image_key] != digest:
                            raise ValueError(f"{uid}: conflicting frozen image for {source.name}")
                        continue
                    seen_images[image_key] = digest
                    target = destination / source.name
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                    unique_images += 1
                    linked_images += 1
                continue
            frames = task.get("frame_paths") or []
            if not 0 < len(frames) <= args.max_frames:
                raise ValueError(f"{uid}: expected 1..{args.max_frames} archived frames, got {len(frames)}")
            video_tasks += 1
            frame_counts[len(frames)] += 1
            video_key = (task["family"], video_path)
            frame_digests = tuple(task["media_sha256"])
            if video_key in seen_videos:
                if seen_videos[video_key] != frame_digests:
                    raise ValueError(f"{uid}: conflicting frozen frame set for {video_path}")
                continue
            seen_videos[video_key] = frame_digests
            unique_videos += 1

            stem = media_stem(video_path)
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

    frozen_media = sorted(path for path in args.out_root.rglob("*") if path.is_file())
    manifest = {
        "manifest_version": 2,
        "max_frames_per_video": args.max_frames,
        "video_task_frame_count_histogram": dict(sorted(frame_counts.items())),
        "accepted_tasks": len(found),
        "image_tasks": image_tasks,
        "unique_images": unique_images,
        "linked_images": linked_images,
        "video_tasks": video_tasks,
        "unique_videos": unique_videos,
        "linked_frames": linked_frames,
        "task_files": {str(path.absolute()): sha256(path) for path in args.tasks},
        "trace_files": {str(path.absolute()): sha256(path) for path in args.traces},
        "task_fingerprints": {
            uid: traces[uid]["task_fingerprint"] for uid in sorted(found)
        },
        "media_sha256": {
            str(path.relative_to(args.out_root)): sha256(path) for path in frozen_media
        },
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
