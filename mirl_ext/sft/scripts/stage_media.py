"""Stage task media for off-cluster trace generation. Teacher-only copies.

Copies each task's images (all of them, original order) next to the task file
and extracts video_frames (config.json) evenly-spaced JPEG frames per video,
first and last frame always included -- the same count build_sft_parquet writes
into the student rows, so teacher and student see the same frames. Filenames
are content-hashed from the source path, so staging is deterministic and
label-free. Rewrites the task JSONL with image_paths/frame_paths pointing at
the staged copies (resolvable on the laptop via --image-root) and stamps
staging_version into every task. Student parquets are never touched.

    python mirl_ext/sft/scripts/stage_media.py --tasks sft_tasks.jsonl --out-root data/sft/media

The frame extractor is the seam for the planned interleaved RGB+dense-ts
format (rl/ts_native_DESIGN.md): v2 crops 1-2 RGB views from the tactile
composite (4 camera views + sensor panel per frame) instead of staging the
full frame, and stages ts chunks alongside. Any such change MUST bump
STAGING_VERSION -- teacher/student frame parity is keyed on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import config_path, iter_jsonl  # noqa: E402
from mirl_ext.data.schema import media_stem as _stem  # noqa: E402
from mirl_ext.sft.artifacts import is_sha256_digest, sha256, task_fingerprint  # noqa: E402

VIDEO_FRAMES = int(config_path("video_frames", "MIRL_VIDEO_FRAMES", "8"))
STAGING_VERSION = f"v4-strict-media-640px-q85-f{VIDEO_FRAMES}"


def _sampled_frame_indices(total: int, n: int) -> tuple[int, ...]:
    """Return deterministic, unique indices spanning the complete video."""
    if n <= 0:
        raise ValueError(f"frame count must be positive, got {n}")
    if total <= 0:
        return ()
    return tuple(sorted({int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)}))


def _cached_frame_names(
    existing: list[Path], expected: list[str], *, stem: str
) -> list[str] | None:
    """Reuse a cache only when it exactly represents the current frame plan."""
    if not existing:
        return None
    names = [path.name for path in existing]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(
            f"stale staged frames for {stem}: found {len(names)}, expected {len(expected)} "
            f"for {STAGING_VERSION}; use a fresh --out-root"
        )
    return expected


def _video_cache_spec(
    src: Path,
    n: int,
    total: int,
    picks: tuple[int, ...],
    frame_names: list[str],
) -> dict:
    stat = src.stat()
    return {
        "staging_version": STAGING_VERSION,
        "source_path": str(src.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "requested_frames": n,
        "total_frames": total,
        "sampled_indices": list(picks),
        "frame_names": frame_names,
    }


def _validate_video_cache(
    existing: list[Path],
    expected: list[str],
    manifest_path: Path,
    spec: dict,
) -> list[str] | None:
    cached = _cached_frame_names(existing, expected, stem=manifest_path.stem.removesuffix(".frames"))
    if cached is None:
        if manifest_path.exists():
            raise ValueError(f"orphaned staged-video manifest: {manifest_path}")
        return None
    if not manifest_path.is_file():
        raise ValueError(
            f"legacy staged frames have no provenance manifest: {manifest_path}; "
            "use a fresh --out-root"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid staged-video manifest: {manifest_path}") from exc
    recorded_spec = {key: manifest.get(key) for key in spec}
    if recorded_spec != spec:
        raise ValueError(
            f"staged-video provenance changed for {manifest_path.stem}; use a fresh --out-root"
        )
    hashes = {path.name: sha256(path) for path in existing}
    if manifest.get("frame_sha256") != hashes:
        raise ValueError(
            f"staged-video frame content changed for {manifest_path.stem}; use a fresh --out-root"
        )
    return cached


def frames_from_video(src: Path, n: int, dest_dir: Path, stem: str) -> list[str]:
    # A path-hashed directory may survive configuration changes. Reuse only an
    # exact current plan; otherwise stop instead of silently mixing 6/8/12-frame
    # representations under one staging version.
    if n <= 0:
        raise ValueError(f"frame count must be positive, got {n}")
    existing = sorted(dest_dir.glob(f"{stem}_f*.jpg"))
    manifest_path = dest_dir / f"{stem}.frames.json"
    import cv2

    cap = cv2.VideoCapture(str(src))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            if existing:
                raise ValueError(f"cannot validate cached frames for unreadable video {src}")
            return []
        # Evenly spaced over [0, total-1]: first and last frames always included.
        picks = _sampled_frame_indices(total, n)
        expected = [f"{stem}_f{k:02d}.jpg" for k in range(len(picks))]
        spec = _video_cache_spec(src, n, total, picks, expected)
        if cached := _validate_video_cache(existing, expected, manifest_path, spec):
            return cached
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
            if cv2.imwrite(str(dest_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                out.append(name)
        if out != expected:
            raise RuntimeError(
                f"incomplete frame extraction for {src}: wrote {len(out)} of {len(expected)}; "
                "use a fresh --out-root before retrying"
            )
        manifest = {
            **spec,
            "frame_sha256": {name: sha256(dest_dir / name) for name in expected},
        }
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)
        return out
    finally:
        cap.release()


def _image_cache_spec(src: Path, max_side: int) -> dict:
    stat = src.stat()
    return {
        "staging_version": STAGING_VERSION,
        "source_path": str(src.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "max_side": max_side,
    }


def stage_image(src: Path, dest_dir: Path, max_side: int) -> str:
    """Stage one image, accepting only a source-matched, hashed cache entry."""
    stem = _stem(str(src))
    manifest_path = dest_dir / f"{stem}.image.json"
    spec = _image_cache_spec(src, max_side)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid staged-image manifest: {manifest_path}") from exc
        if {key: manifest.get(key) for key in spec} != spec:
            raise ValueError(
                f"staged-image provenance changed for {src}; use a fresh --out-root"
            )
        name = manifest.get("output_name")
        dest = dest_dir / str(name)
        if not name or not dest.is_file() or sha256(dest) != manifest.get("output_sha256"):
            raise ValueError(
                f"staged-image content changed for {src}; use a fresh --out-root"
            )
        return str(name)

    legacy = [path for path in dest_dir.glob(f"{stem}.*") if path != manifest_path]
    if legacy:
        raise ValueError(
            f"legacy staged image has no provenance manifest for {src}; use a fresh --out-root"
        )

    name = f"{stem}{src.suffix.lower()}"
    dest = dest_dir / name
    if not max_side:
        shutil.copy2(src, dest)
    else:
        import cv2

        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"cannot decode image {src}")
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            name = f"{stem}.jpg"
            dest = dest_dir / name
            if not cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"failed to write staged image {dest}")
        else:
            shutil.copy2(src, dest)

    manifest = {
        **spec,
        "output_name": name,
        "output_sha256": sha256(dest),
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
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

    tasks = list(iter_jsonl(args.tasks))
    missing_fingerprints = [
        task.get("uid", "<missing uid>")
        for task in tasks
        if not is_sha256_digest(task.get("source_row_fingerprint"))
    ]
    if missing_fingerprints:
        preview = ", ".join(map(str, missing_fingerprints[:8]))
        raise SystemExit(
            f"{len(missing_fingerprints)} task(s) lack source-row provenance; "
            f"re-run export_sft_tasks.py (first: {preview})"
        )
    missing_media = sorted(
        {
            raw
            for task in tasks
            for raw in [*(task.get("image_paths") or []), task.get("video_path")]
            if raw and not Path(raw).is_file()
        }
    )
    if missing_media:
        preview = ", ".join(missing_media[:8])
        raise FileNotFoundError(
            f"{len(missing_media)} task media source(s) are missing; first: {preview}"
        )
    n_img = n_vid = 0

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

    media_hashes: dict[str, str] = {}
    for task in tasks:
        dest_dir = args.out_root / task["family"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        staged_images = []
        for raw in task.get("image_paths") or []:
            p = Path(raw)
            name = stage_image(p, dest_dir, args.max_side)
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
                raise ValueError(f"no frames extracted for {task['video_path']}")
        task["staging_version"] = f"{STAGING_VERSION}-im{args.max_side}"
        teacher_media = task.get("frame_paths") or task.get("image_paths") or []
        for path in teacher_media:
            if path not in media_hashes:
                media_hashes[path] = sha256(Path(path))
        task["media_sha256"] = [media_hashes[path] for path in teacher_media]
        task["task_fingerprint"] = task_fingerprint(task)
    staged = args.tasks.with_suffix(".staged.jsonl")
    staged.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    size = sum(f.stat().st_size for f in args.out_root.rglob("*") if f.is_file())
    print(f"images={n_img} videos={n_vid} media={size / 1e6:.0f}MB")
    print(f"-> {staged} (copy it + {args.out_root}/ to the laptop)")


if __name__ == "__main__":
    main()
