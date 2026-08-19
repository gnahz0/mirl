"""Stage task media for off-cluster trace generation. Teacher-only copies.

Copies each task's images (all of them, original order) next to the task file
and extracts video_frames (config.json) evenly-spaced JPEG frames per video,
first and last frame always included -- the same count build_sft_parquet writes
into the student rows, so teacher and student see the same frames. Filenames
are content-hashed from the source path, so staging is deterministic and
label-free. Rewrites the task JSONL with image_paths/frame_paths pointing at
the staged copies (resolvable on the laptop via --image-root) and stamps
staging_version into every task. Student parquets are never touched.

    python mirl_ext/sft/stage_media.py --tasks sft_tasks.jsonl --out-root data/sft/media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.schema import config_path  # noqa: E402

STAGING_VERSION = "v2-640px-q85"
VIDEO_FRAMES = int(config_path("video_frames", "MIRL_VIDEO_FRAMES", "8"))


def _stem(src: str) -> str:
    return hashlib.sha1(src.encode()).hexdigest()[:20]


def frames_from_video(src: Path, n: int, dest_dir: Path, stem: str) -> list[str]:
    import cv2

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    # Evenly spaced over [0, total-1]: first and last frames always included.
    picks = sorted({int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)})
    out = []
    for k, idx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # Bound the long side so a frame stays ~tens of KB.
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        name = f"{stem}_f{k:02d}.jpg"
        cv2.imwrite(str(dest_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(name)
    cap.release()
    return out


def stage_image(src: Path, dest_dir: Path, max_side: int) -> str | None:
    """Copy (or, with --max-side, bound) one image; returns the staged name."""
    name = f"{_stem(str(src))}{src.suffix.lower()}"
    dest = dest_dir / name
    if dest.exists():
        return name
    if not max_side:
        shutil.copy(src, dest)
        return name
    import cv2

    img = cv2.imread(str(src))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        name = f"{_stem(str(src))}.jpg"
        cv2.imwrite(str(dest_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        shutil.copy(src, dest)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True, help="media lands in <out-root>/<family>/")
    ap.add_argument("--max-side", type=int, default=0,
                    help="if >0, bound staged images to this long side (teacher copies "
                    "only; use when full-size images overflow the API payload limit)")
    ap.add_argument("--workers", type=int, default=8, help="parallel video extractions")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    n_img = n_vid = n_miss = 0

    # Many rows share one recording: extract each unique video once.
    from concurrent.futures import ProcessPoolExecutor

    jobs = {t["video_path"]: args.out_root / t["family"] for t in tasks if t.get("video_path")}
    for dest in set(jobs.values()):
        dest.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            src: pool.submit(frames_from_video, Path(src), VIDEO_FRAMES, dest, _stem(src))
            for src, dest in jobs.items()
            if Path(src).is_file()
        }
        frame_names = {src: fut.result() for src, fut in futs.items()}

    for task in tasks:
        dest_dir = args.out_root / task["family"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        staged_images = []
        for raw in task.get("image_paths") or []:
            p = Path(raw)
            if not p.is_file():
                n_miss += 1
                continue
            name = stage_image(p, dest_dir, args.max_side)
            if name:
                staged_images.append(str(dest_dir / name))
        if staged_images:
            task["image_paths"] = staged_images
            n_img += 1
        if task.get("video_path"):
            names = frame_names.get(task["video_path"]) or []
            if names:
                task["frame_paths"] = [str(dest_dir / name) for name in names]
                n_vid += 1
            else:
                n_miss += 1
        task["staging_version"] = STAGING_VERSION
    staged = args.tasks.with_suffix(".staged.jsonl")
    staged.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    size = sum(f.stat().st_size for f in args.out_root.rglob("*") if f.is_file())
    print(f"images={n_img} videos={n_vid} missing={n_miss} media={size / 1e6:.0f}MB")
    print(f"-> {staged} (copy it + {args.out_root}/ to the laptop)")


if __name__ == "__main__":
    main()
