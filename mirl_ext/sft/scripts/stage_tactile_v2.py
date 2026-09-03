"""Stage tactile tasks as sampled synchronized RGB+tactile composite frames.

Each source mp4 already contains the representation used by MMTouch: multiple
RGB camera views on the left and tactile pressure heatmaps on the right. Keep
that complete composite so the teacher can jointly reason about the scene and
pressure. Sample approximately 1 FPS over the complete (at most 24-second)
clip, with at least four distinct frames for short clips. First and final
frames are always included.

    srun -p cpu -c 8 --mem=48G <env>/bin/python \
        mirl_ext/sft/scripts/stage_tactile_v2.py --tasks sft_tasks.jsonl \
        --out-root $SCRATCH/data/sft_media_v2 --max-recordings 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import iter_jsonl, media_stem  # noqa: E402
from mirl_ext.sft.artifacts import sha256, task_fingerprint  # noqa: E402

STAGING_VERSION = "v10-tactile-composite-1fps-min4-trunc24s-640px-q85"
MIN_FRAMES = 4
MAX_FRAMES = 24
MAX_DURATION_S = 24.0
MAX_SIDE = 640
PT_MAP_NAMES = ("haptic_ts_train.parquet",)


def sampled_frame_indices(total_frames: int, fps: float) -> tuple[int, ...]:
    """Approximately 1 FPS, min 4, max 24, uniformly spanning <=24 seconds."""
    if total_frames <= 0:
        raise ValueError(f"video frame count must be positive, got {total_frames}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"video FPS must be finite and positive, got {fps}")

    window_frames = min(total_frames, int(math.ceil(MAX_DURATION_S * fps)))
    if window_frames < MIN_FRAMES:
        raise ValueError(f"need {MIN_FRAMES} distinct video frames, found {window_frames}")
    duration_s = window_frames / fps
    count = min(MAX_FRAMES, max(MIN_FRAMES, int(math.ceil(duration_s))))
    indices = tuple(int(round(i * (window_frames - 1) / (count - 1))) for i in range(count))
    if len(set(indices)) != len(indices):
        raise ValueError("sampling plan did not produce distinct frames")
    return indices


def stem_to_pt(map_files: list[str]) -> dict[str, tuple[str, str | None]]:
    """Recording stem -> tactile tensor reference (used by truncation tooling)."""
    import pyarrow.parquet as pq

    out: dict[str, tuple[str, str | None]] = {}
    for f in map_files:
        pf = pq.ParquetFile(f)
        if "signals" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(batch_size=128, columns=["signals"]):
            for sigs in batch.column("signals").to_pylist():
                for s in sigs or []:
                    is_tactile = (
                        isinstance(s, dict)
                        and str(s.get("signal", "")).endswith(".pt")
                        and s.get("format") == "tactile_pt"
                    )
                    if is_tactile:
                        out[Path(s["signal"]).stem] = (s["signal"], s.get("key"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True, help="export_sft_tasks output (all families)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--max-recordings", type=int, default=0, help="pilot cap; 0 = all")
    args = ap.parse_args()

    from torchcodec.decoders import VideoDecoder

    tasks = [t for t in iter_jsonl(args.tasks) if t.get("family") == "tactile_train"]
    print(f"{len(tasks)} tactile tasks")

    media_dir = args.out_root / "tactile_composite"
    media_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, list[str] | None] = {}
    media_hashes: dict[str, str] = {}
    out_tasks, skipped = [], {"no_video": 0, "error": 0}

    for task in tasks:
        video = task.get("video_path") or ""
        if not video:
            skipped["no_video"] += 1
            continue
        vstem = Path(video).stem
        if video not in staged:
            if args.max_recordings and sum(v is not None for v in staged.values()) >= args.max_recordings:
                staged[video] = None
            else:
                staged[video] = stage_recording(video, vstem, media_dir, skipped, VideoDecoder)
        frames = staged[video]
        if not frames:
            continue
        task = dict(task)
        task["frame_paths"] = frames
        task["image_paths"] = []
        task["staging_version"] = STAGING_VERSION
        for p in frames:
            if p not in media_hashes:
                media_hashes[p] = sha256(Path(p))
        task["media_sha256"] = [media_hashes[p] for p in frames]
        task["task_fingerprint"] = task_fingerprint(task)
        out_tasks.append(task)

    out_path = args.tasks.with_name(args.tasks.stem + ".tactile_v2.jsonl")
    with out_path.open("w") as fh:
        for t in out_tasks:
            fh.write(json.dumps(t) + "\n")
    n_rec = sum(1 for v in staged.values() if v)
    print(f"staged {n_rec} recordings, {len(out_tasks)} tasks -> {out_path}\nskipped: {skipped}")


def stage_recording(video, vstem, media_dir, skipped, VideoDecoder):
    from PIL import Image

    try:
        dec = VideoDecoder(video)
        meta = dec.metadata
        indices = sampled_frame_indices(meta.num_frames, meta.average_fps)
        stem20 = media_stem(video)
        paths = []
        for k, frame_index in enumerate(indices):
            sec = frame_index / meta.average_fps
            frame = dec[frame_index].permute(1, 2, 0).numpy()
            image = Image.fromarray(frame)
            if max(image.size) > MAX_SIDE:
                image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            frame_path = media_dir / (f"{stem20}_f{k:02d}_t{int(round(sec * 1000)):06d}.jpg")
            image.save(frame_path, quality=85)
            paths.append(str(frame_path))
        return paths
    except Exception as e:
        skipped["error"] += 1
        print(f"  ERR {vstem}: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    main()
