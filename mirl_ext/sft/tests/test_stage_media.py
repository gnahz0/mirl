"""Pure-stdlib tests for deterministic, fail-closed video staging caches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mirl_ext.sft.artifacts import (
    sha256,
    task_fingerprint,
    verify_audit_manifest,
    verify_frozen_media_manifest,
    verify_frozen_selection,
    verify_media_hashes,
)
from mirl_ext.sft.scripts.stage_media import (
    _cached_frame_names,
    _sampled_frame_indices,
    _validate_video_cache,
    _video_cache_spec,
    stage_image,
)
from mirl_ext.sft.scripts.stage_tactile_v2 import (
    MAX_FRAMES,
    MIN_FRAMES,
    PT_MAP_NAMES,
    sampled_frame_indices,
)


class StageMediaCacheTest(unittest.TestCase):
    def test_tactile_v2_uses_an_explicit_signal_index(self) -> None:
        self.assertEqual(PT_MAP_NAMES, ("haptic_ts_train.parquet",))

    def test_sampling_spans_video_and_deduplicates_short_clips(self) -> None:
        self.assertEqual(_sampled_frame_indices(100, 8)[0], 0)
        self.assertEqual(_sampled_frame_indices(100, 8)[-1], 99)
        self.assertEqual(len(_sampled_frame_indices(100, 8)), 8)
        self.assertEqual(_sampled_frame_indices(3, 8), (0, 1, 2))
        with self.assertRaisesRegex(ValueError, "positive"):
            _sampled_frame_indices(100, 0)

    def test_tactile_sampling_is_one_fps_with_four_frame_floor(self) -> None:
        cases = (
            (48, 30.0, MIN_FRAMES, 47),  # 1.6 s: denser than 1 FPS
            (90, 30.0, MIN_FRAMES, 89),  # 3.0 s: denser than 1 FPS
            (177, 30.0, 6, 176),  # 5.9 s: approximately 1 FPS
            (180, 30.0, 6, 179),  # 6.0 s: exactly 1 FPS
            (720, 30.0, MAX_FRAMES, 719),  # 24 s: one frame per second
            (4606, 30.0, MAX_FRAMES, 719),  # defensive 24-second cap
        )
        for total_frames, fps, expected_count, expected_last in cases:
            with self.subTest(total_frames=total_frames, fps=fps):
                indices = sampled_frame_indices(total_frames, fps)
                self.assertEqual(len(indices), expected_count)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], expected_last)
                self.assertEqual(len(set(indices)), len(indices))
                self.assertEqual(tuple(sorted(indices)), indices)

    def test_tactile_sampling_rejects_invalid_or_too_short_video(self) -> None:
        for total_frames, fps in ((0, 30.0), (30, 0.0), (30, float("nan"))):
            with self.subTest(total_frames=total_frames, fps=fps):
                with self.assertRaisesRegex(ValueError, "positive"):
                    sampled_frame_indices(total_frames, fps)
        with self.assertRaisesRegex(ValueError, "4 distinct"):
            sampled_frame_indices(3, 30.0)

    def test_cache_reuse_requires_the_exact_current_plan(self) -> None:
        expected = [f"abc_f{index:02d}.jpg" for index in range(8)]
        exact = [Path(name) for name in expected]

        self.assertEqual(_cached_frame_names(exact, expected, stem="abc"), expected)
        self.assertIsNone(_cached_frame_names([], expected, stem="abc"))
        with self.assertRaisesRegex(ValueError, "found 12, expected 8"):
            _cached_frame_names(
                [Path(f"abc_f{index:02d}.jpg") for index in range(12)],
                expected,
                stem="abc",
            )

    def test_video_cache_requires_matching_manifest_and_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source.mp4"
            source.write_bytes(b"source-v1")
            names = [f"abc_f{index:02d}.jpg" for index in range(2)]
            frames = [root / name for name in names]
            for index, frame in enumerate(frames):
                frame.write_bytes(f"frame-{index}".encode())
            manifest_path = root / "abc.frames.json"
            picks = _sampled_frame_indices(10, 2)
            spec = _video_cache_spec(source, 2, 10, picks, names)

            with self.assertRaisesRegex(ValueError, "no provenance manifest"):
                _validate_video_cache(frames, names, manifest_path, spec)

            manifest_path.write_text(
                json.dumps(
                    {
                        **spec,
                        "frame_sha256": {frame.name: sha256(frame) for frame in frames},
                    }
                )
            )
            self.assertEqual(_validate_video_cache(frames, names, manifest_path, spec), names)

            source.write_bytes(b"source-v2-is-different")
            changed_spec = _video_cache_spec(source, 2, 10, picks, names)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                _validate_video_cache(frames, names, manifest_path, changed_spec)

    def test_task_fingerprint_tracks_teacher_input_but_not_answer_key(self) -> None:
        task = {
            "uid": "tactile_train#0",
            "family": "tactile_train",
            "row_index": 0,
            "data_source": "initial_fingers",
            "prompt": "<video> Which fingers?",
            "frame_paths": ["abc_f00.jpg"],
            "staging_version": "v3",
            "media_sha256": ["deadbeef"],
            "ground_truth": "A,B",
        }
        fingerprint = task_fingerprint(task)
        self.assertEqual(fingerprint, task_fingerprint({**task, "ground_truth": "F"}))
        self.assertNotEqual(fingerprint, task_fingerprint({**task, "prompt": "changed"}))

    def test_training_files_must_match_audit_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            parquet = root / "tactile_train_sft.parquet"
            parquet.write_bytes(b"audited parquet bytes")
            media = root / "frame.jpg"
            media.write_bytes(b"audited training frame")
            frozen_manifest = root / "frozen_manifest.json"
            frozen_manifest.write_bytes(b"audited frozen manifest")
            (root / "audit_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 2,
                        "parquet_sha256": {parquet.name: sha256(parquet)},
                        "frozen_media_manifest": {
                            "path": str(frozen_manifest),
                            "sha256": sha256(frozen_manifest),
                        },
                        "training_media_sha256": {str(media.absolute()): sha256(media)},
                    }
                )
            )
            verify_audit_manifest(root, [parquet])

            parquet.write_bytes(b"changed after audit")
            with self.assertRaisesRegex(ValueError, "differs from audit"):
                verify_audit_manifest(root, [parquet])

            parquet.write_bytes(b"audited parquet bytes")
            media.write_bytes(b"changed media after audit")
            with self.assertRaisesRegex(ValueError, "training media differs"):
                verify_audit_manifest(root, [parquet])

    def test_frozen_media_manifest_covers_and_hashes_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            family = root / "tactile_train"
            family.mkdir()
            frame = family / "abc_f00.jpg"
            frame.write_bytes(b"frame-v1")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 2,
                        "media_sha256": {str(frame.relative_to(root)): sha256(frame)},
                    }
                )
            )
            self.assertEqual(verify_frozen_media_manifest(root), root / "manifest.json")

            frame.write_bytes(b"frame-v2")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                verify_frozen_media_manifest(root)

    def test_image_cache_requires_source_recipe_and_output_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source.png"
            destination = root / "staged"
            destination.mkdir()
            source.write_bytes(b"not-decoded-when-max-side-is-zero")

            name = stage_image(source, destination, 0)
            self.assertEqual(stage_image(source, destination, 0), name)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                stage_image(source, destination, 640)

            source.write_bytes(b"changed-source")
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                stage_image(source, destination, 0)

    def test_frozen_media_is_bound_to_trace_bytes_and_task_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            trace_file = Path(raw_root) / "traces.jsonl"
            trace_file.write_text('{"uid":"x","status":"accepted"}\n')
            traces = {"x": {"task_fingerprint": "a" * 64}}
            manifest = {
                "trace_files": {str(trace_file.absolute()): sha256(trace_file)},
                "task_fingerprints": {"x": "a" * 64},
            }
            verify_frozen_selection(manifest, traces, [trace_file])

            trace_file.write_text('{"uid":"x","status":"accepted","changed":true}\n')
            with self.assertRaisesRegex(ValueError, "different trace bytes"):
                verify_frozen_selection(manifest, traces, [trace_file])
            trace_file.write_text('{"uid":"x","status":"accepted"}\n')
            with self.assertRaisesRegex(ValueError, "task fingerprints"):
                verify_frozen_selection(
                    manifest,
                    {"x": {"task_fingerprint": "b" * 64}},
                    [trace_file],
                )

    def test_teacher_media_bytes_must_match_staged_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media = Path(raw_root) / "frame.jpg"
            media.write_bytes(b"teacher frame")
            expected = [sha256(media)]
            verify_media_hashes([media], expected)

            media.write_bytes(b"corrupted copy")
            with self.assertRaisesRegex(ValueError, "differs from task fingerprint"):
                verify_media_hashes([media], expected)


if __name__ == "__main__":
    unittest.main()
