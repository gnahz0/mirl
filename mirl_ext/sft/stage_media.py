"""Stage task media for off-cluster trace generation.

Copies each task's image next to the task file and extracts N evenly-spaced
JPEG frames per video (teacher endpoints take images, not mp4s; shipping frames
is ~100x lighter than shipping videos). Rewrites the task JSONL with
`image_path`/`frame_paths` pointing at the staged copies, resolvable on the
laptop via --image-root.

    # on the cluster (cv2 lives in the alec-mv env)
    python mirl_ext/sft/stage_media.py --tasks iv_tasks.jsonl --out-root data/sft/media
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def frames_from_video(src: Path, n: int, dest_dir: Path, stem: str) -> list[str]:
    import cv2

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True, help="media lands in <out-root>/<family>/")
    ap.add_argument("--frames", type=int, default=6, help="frames sampled per video")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    n_img = n_vid = n_miss = 0
    for task in tasks:
        dest_dir = args.out_root / task["family"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Content-addressed stem: stable across runs, no collisions across sources.
        src = task.get("image_path") or task.get("video_path") or ""
        stem = hashlib.sha1(src.encode()).hexdigest()[:20]
        if task.get("image_path"):
            p = Path(task["image_path"])
            if not p.is_file():
                n_miss += 1
                continue
            name = f"{stem}{p.suffix.lower()}"
            if not (dest_dir / name).exists():
                shutil.copy(p, dest_dir / name)
            task["image_path"] = str(dest_dir / name)
            n_img += 1
        elif task.get("video_path"):
            p = Path(task["video_path"])
            if not p.is_file():
                n_miss += 1
                continue
            names = frames_from_video(p, args.frames, dest_dir, stem)
            if not names:
                n_miss += 1
                continue
            task["frame_paths"] = [str(dest_dir / n) for n in names]
            n_vid += 1

    staged = args.tasks.with_suffix(".staged.jsonl")
    staged.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    size = sum(f.stat().st_size for f in args.out_root.rglob("*") if f.is_file())
    print(f"images={n_img} videos={n_vid} missing={n_miss} media={size/1e6:.0f}MB")
    print(f"-> {staged} (copy it + {args.out_root}/ to the laptop)")


if __name__ == "__main__":
    main()
