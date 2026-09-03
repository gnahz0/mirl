"""Fail-closed integrity audit for a built MIRL SFT parquet directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.sft.artifacts import (  # noqa: E402
    frame_index,
    sha256,
    verify_frozen_media_manifest,
    verify_frozen_selection,
)
from mirl_ext.sft.traces import accepted_traces  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="override the frozen manifest's frame limit",
    )
    parser.add_argument("--smoke-train-prefix", type=int, default=256)
    parser.add_argument("--val-prefix", type=int, default=512)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    traces = accepted_traces(args.traces)
    frozen_manifest_path = verify_frozen_media_manifest(args.media_root)
    frozen_manifest = json.loads(frozen_manifest_path.read_text())
    manifest_max_frames = frozen_manifest.get("max_frames_per_video")
    max_frames = args.max_frames if args.max_frames is not None else manifest_max_frames
    if not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError(f"invalid maximum frame count: {max_frames}")
    verify_frozen_selection(frozen_manifest, traces, args.traces)
    files = sorted(args.parquet_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets under {args.parquet_root}")

    uids: dict[str, str] = {}
    split_flags: dict[str, list[bool]] = {"train": [], "val": []}
    modalities = {"train": {"image": 0, "video": 0}, "val": {"image": 0, "video": 0}}
    media_keys: dict[str, set[tuple]] = {"train": set(), "val": set()}
    image_paths: set[Path] = set()
    frame_paths: set[Path] = set()
    staged: dict[str, dict[str, list[str]]] = {}  # family -> frozen frame index

    for path in files:
        split = "val" if path.name.endswith("_sft_val.parquet") else "train"
        for row in pq.read_table(path).to_pylist():
            info = json.loads(row["extra_info"])
            uid = info["uid"]
            if uid in uids:
                raise ValueError(f"duplicate uid {uid}: {uids[uid]} and {path.name}")
            uids[uid] = path.name

            trace = traces.get(uid)
            if trace is None:
                raise ValueError(f"{uid}: no accepted source trace")
            if info.get("ground_truth") != trace.get("ground_truth"):
                raise ValueError(f"{uid}: parquet/trace ground-truth mismatch")
            if info.get("source_row_fingerprint") != trace.get("source_row_fingerprint"):
                raise ValueError(f"{uid}: parquet/trace source-row-fingerprint mismatch")
            if info.get("staging_version") != trace.get("staging_version"):
                raise ValueError(f"{uid}: parquet/trace staging-version mismatch")
            if info.get("task_fingerprint") != trace.get("task_fingerprint"):
                raise ValueError(f"{uid}: parquet/trace task-fingerprint mismatch")
            if info.get("mode") not in {"answer_blind_zero_shot", "answer_conditioned"}:
                raise ValueError(f"{uid}: unexpected generation mode {info.get('mode')!r}")
            if row["messages"][-1] != {"role": "assistant", "content": trace["response"]}:
                raise ValueError(f"{uid}: assistant response differs from accepted trace")

            images = row.get("images") or []
            videos = row.get("videos") or []
            if bool(images) == bool(videos):
                raise ValueError(f"{uid}: expected exactly one of images/videos")
            text = "".join(message["content"] for message in row["messages"])
            if text.count("<image>") != len(images) or text.count("<video>") != len(videos):
                raise ValueError(f"{uid}: media placeholder mismatch")

            if images:
                modalities[split]["image"] += 1
                paths = tuple(sorted(str(entry["image"]) for entry in images))
                image_paths.update(map(Path, paths))
                media_keys[split].add(("image", *paths))
                is_video = False
            else:
                modalities[split]["video"] += 1
                if len(videos) != 1:
                    raise ValueError(f"{uid}: expected one video entry, got {len(videos)}")
                frames = videos[0]["video"]
                if not isinstance(frames, list) or not 0 < len(frames) <= max_frames:
                    raise ValueError(f"{uid}: invalid staged frame list length {len(frames)}")
                stems = {Path(frame).name.rsplit("_f", 1)[0] for frame in frames}
                if len(stems) != 1:
                    raise ValueError(f"{uid}: frame list mixes video stems")
                stem = next(iter(stems))
                family = uid.split("#", 1)[0]
                if family not in staged:
                    staged[family] = frame_index(args.media_root / family)
                expected = staged[family].get(stem, [])
                if frames != expected:
                    raise ValueError(
                        f"{uid}: parquet has {len(frames)} frames but frozen index has {len(expected)}"
                    )
                frame_paths.update(map(Path, frames))
                media_keys[split].add(("video", family, stem))
                is_video = True
            split_flags[split].append(is_video)

    missing_traces = set(traces) - set(uids)
    if missing_traces:
        preview = ", ".join(sorted(missing_traces)[:8])
        raise ValueError(f"{len(missing_traces)} accepted traces missing from parquets; first: {preview}")
    media_root = args.media_root.absolute()
    bad_images = [
        path
        for path in image_paths
        if not path.is_file() or not path.absolute().is_relative_to(media_root)
    ]
    if bad_images:
        raise FileNotFoundError(
            f"{len(bad_images)} missing/out-of-root images; first: {bad_images[0]}"
        )
    bad_frames = [
        path
        for path in frame_paths
        if not path.is_file() or not path.absolute().is_relative_to(media_root)
    ]
    if bad_frames:
        raise FileNotFoundError(f"{len(bad_frames)} missing/out-of-root frames; first: {bad_frames[0]}")
    shared_media = media_keys["train"] & media_keys["val"]
    if shared_media:
        raise ValueError(f"{len(shared_media)} media groups straddle train/val; first: {next(iter(shared_media))}")

    train_prefix = split_flags["train"][: args.smoke_train_prefix]
    val_prefix = split_flags["val"][: args.val_prefix]
    prefix_counts = {
        "train": {"image": len(train_prefix) - sum(train_prefix), "video": sum(train_prefix)},
        "val": {"image": len(val_prefix) - sum(val_prefix), "video": sum(val_prefix)},
    }
    training_media = {
        path.absolute() for path in image_paths | frame_paths
    }
    manifest = {
        "manifest_version": 2,
        "rows": {split: len(flags) for split, flags in split_flags.items()},
        "modalities": modalities,
        "accepted_trace_rows": len(traces),
        "unique_uids": len(uids),
        "train_val_shared_media_groups": 0,
        "prefix_modalities": prefix_counts,
        "parquet_sha256": {path.name: sha256(path) for path in files},
        "frozen_media_manifest": {
            "path": str(frozen_manifest_path.absolute()),
            "sha256": sha256(frozen_manifest_path),
        },
        "training_media_sha256": {
            str(path): sha256(path) for path in sorted(training_media)
        },
    }
    manifest_path = args.parquet_root / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
