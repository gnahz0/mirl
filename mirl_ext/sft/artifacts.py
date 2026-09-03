"""Small filesystem helpers shared by SFT artifact-building commands."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

_TASK_FINGERPRINT_FIELDS = (
    "uid",
    "family",
    "row_index",
    "source_row_fingerprint",
    "data_source",
    "prompt",
    "image_paths",
    "video_path",
    "frame_paths",
    "staging_version",
    "media_sha256",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_sha256_digest(value: Any) -> bool:
    """Return whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
        and value == value.lower()
    )


def source_row_fingerprint(row: Mapping[str, Any], family: str, row_index: int) -> str:
    """Hash ordered, answer-blind source fields that become the student input."""
    payload = {
        "fingerprint_version": 1,
        "family": family,
        "row_index": row_index,
        "data_source": row.get("data_source"),
        "prompt": row.get("prompt"),
        "images": row.get("images") or [],
        "videos": row.get("videos") or [],
    }
    return _canonical_sha256(payload)


class FrozenMediaIndex(NamedTuple):
    images: dict[str, str]
    videos: dict[str, list[str]]


def frozen_media_index(staged_dir: Path) -> FrozenMediaIndex:
    """Index frozen images and video-frame lists with one directory scan."""
    images: dict[str, str] = {}
    videos: dict[str, list[str]] = {}
    if staged_dir.is_dir():
        for path in sorted(staged_dir.iterdir()):
            if path.name.endswith(".jpg") and "_f" in path.name:
                videos.setdefault(path.name.rsplit("_f", 1)[0], []).append(str(path))
            elif path.is_file() and path.suffix != ".json":
                if path.stem in images:
                    raise ValueError(f"duplicate frozen image stem under {staged_dir}: {path.stem}")
                images[path.stem] = str(path)
    return FrozenMediaIndex(images=images, videos=videos)


def frame_index(staged_dir: Path) -> dict[str, list[str]]:
    """Compatibility view of the frozen video-frame index."""
    return frozen_media_index(staged_dir).videos


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Hash the answer-blind teacher input and its staged-media identities."""
    payload = {field: task.get(field) for field in _TASK_FINGERPRINT_FIELDS}
    return _canonical_sha256(payload)


def training_media_paths(row: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return every image/frame path that the SFT dataset will open."""
    paths: list[Path] = []
    for entry in row.get("images") or []:
        raw = entry.get("image") if isinstance(entry, Mapping) else entry
        if not raw:
            raise ValueError(f"invalid SFT image entry: {entry!r}")
        paths.append(Path(raw))
    for entry in row.get("videos") or []:
        raw = entry.get("video") if isinstance(entry, Mapping) else entry
        values = raw if isinstance(raw, list) else [raw]
        if not values or any(not value for value in values):
            raise ValueError(f"invalid SFT video entry: {entry!r}")
        paths.extend(Path(value) for value in values)
    return tuple(paths)


def verify_media_hashes(
    paths: Iterable[Path],
    expected_hashes: Iterable[str],
    cache: dict[str, str] | None = None,
) -> None:
    """Verify copied teacher-media bytes, memoizing shared recording paths."""
    media = tuple(Path(path) for path in paths)
    expected = tuple(expected_hashes)
    if len(media) != len(expected):
        raise ValueError(
            f"teacher media/hash count mismatch: {len(media)} paths vs {len(expected)} hashes"
        )
    digests = cache if cache is not None else {}
    for path, wanted in zip(media, expected, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"missing staged teacher media: {path}")
        key = str(path.resolve())
        if key not in digests:
            digests[key] = sha256(path)
        if digests[key] != wanted:
            raise ValueError(f"staged teacher media differs from task fingerprint: {path}")


def verify_frozen_media_manifest(media_root: Path) -> Path:
    """Verify the selector's manifest covers every frozen media byte-for-byte."""
    manifest_path = media_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen-media manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen-media manifest: {manifest_path}") from exc
    if manifest.get("manifest_version") != 2:
        raise ValueError(f"unsupported frozen-media manifest version: {manifest_path}")
    recorded = manifest.get("media_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(f"frozen-media manifest has no media hashes: {manifest_path}")
    actual = {
        str(path.relative_to(media_root)): path
        for path in sorted(media_root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if set(recorded) != set(actual):
        raise ValueError(
            f"frozen-media file set differs from manifest under {media_root}"
        )
    for relative_path, path in actual.items():
        if sha256(path) != recorded[relative_path]:
            raise ValueError(f"frozen media differs from manifest: {path}")
    return manifest_path


def verify_frozen_selection(
    manifest: Mapping[str, Any],
    traces: Mapping[str, Mapping[str, Any]],
    trace_files: Iterable[Path],
) -> None:
    """Bind a frozen-media tree to the exact accepted tasks and trace bytes."""
    expected_trace_files = {
        str(path.absolute()): sha256(path) for path in trace_files
    }
    if manifest.get("trace_files") != expected_trace_files:
        raise ValueError(
            "frozen media were selected from different trace bytes; "
            "re-run select_trace_frames.py into a fresh --out-root"
        )
    expected_task_fingerprints = {
        uid: trace.get("task_fingerprint") for uid, trace in sorted(traces.items())
    }
    if manifest.get("task_fingerprints") != expected_task_fingerprints:
        raise ValueError(
            "frozen media do not match the accepted trace task fingerprints; "
            "re-run select_trace_frames.py into a fresh --out-root"
        )


def verify_audit_manifest(parquet_root: Path, files: Iterable[Path]) -> dict:
    """Verify the complete audited parquet, frame-manifest, and media bytes."""
    manifest_path = parquet_root / "audit_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; run audit_sft_parquet.py before training"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SFT audit manifest: {manifest_path}") from exc
    if manifest.get("manifest_version") != 2:
        raise ValueError(f"unsupported SFT audit manifest version in {manifest_path}")
    recorded = manifest.get("parquet_sha256")
    if not isinstance(recorded, dict):
        raise ValueError(f"SFT audit manifest has no parquet hashes: {manifest_path}")
    selected = tuple(Path(raw_path) for raw_path in files)
    selected_names = {path.name for path in selected}
    if selected_names != set(recorded):
        raise ValueError(
            "selected SFT parquet set differs from the audited set: "
            f"selected={sorted(selected_names)}, audited={sorted(recorded)}"
        )
    for path in selected:
        expected_path = parquet_root / path.name
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"parquet is outside audited root {parquet_root}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"missing audited SFT parquet: {path}")
        digest = sha256(path)
        if recorded.get(path.name) != digest:
            raise ValueError(
                f"SFT parquet differs from audit manifest: {path}; re-run the audit"
            )

    frozen_manifest = manifest.get("frozen_media_manifest")
    if not isinstance(frozen_manifest, dict):
        raise ValueError(f"SFT audit manifest has no frozen-media provenance: {manifest_path}")
    frozen_path = Path(frozen_manifest.get("path", ""))
    if not frozen_path.is_file() or sha256(frozen_path) != frozen_manifest.get("sha256"):
        raise ValueError(
            f"frozen-media manifest differs from SFT audit: {frozen_path}; re-run the audit"
        )

    media_hashes = manifest.get("training_media_sha256")
    if not isinstance(media_hashes, dict) or not media_hashes:
        raise ValueError(f"SFT audit manifest has no training-media hashes: {manifest_path}")
    for raw_path, expected_digest in media_hashes.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing audited SFT training media: {path}")
        if sha256(path) != expected_digest:
            raise ValueError(
                f"SFT training media differs from audit manifest: {path}; re-run the audit"
            )
    return manifest
