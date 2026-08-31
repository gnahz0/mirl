#!/usr/bin/env python3
"""Rewrite the ts families' GRPO parquets to the native pseudo-video path.

Per row: join the plot-image GRPO row to its raw signal by row index in
$DATA/trainedve_raw (100% ground-truth-aligned; asserted per row), render the
Stage-1 frames via mirl_ext.data.signals, quantize to one vertically stacked
grayscale strip PNG per signal (u8 = round((x+1)*127.5) — the exact inverse of
Qwen3.5's 0.5/0.5 rescale+normalize), and emit <family>_<split>_tsnative.parquet
whose only differences from the source are <image> -> <video> and the media
entry. MIRLDataset expands the ``_stack{T}.png`` suffix back into frames at
load time. Rationale, identity-resize conditions, and deviations from Stage-1:
ts_native_DESIGN.md.

    # full build (cluster, under srun)
    python mirl_ext/rl/build_ts_native_parquet.py
    # 3 rows/family: human-inspectable PNGs + processor equivalence check
    python mirl_ext/rl/build_ts_native_parquet.py --probe /scratch/.../ts_native_probe
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mirl_ext.data.schema import DATA_ROOT, SCRATCH_ROOT  # noqa: E402
from mirl_ext.data.signals import family_frames, load_signal  # noqa: E402

CELL = 32  # ViT patch 16 x spatial merge 2: the Stage-1 tile width and tactile side
FAMILIES = ("smellnet", "ecg", "haptic_ts")


def render_frames(sig_entry: dict, max_frames: int) -> tuple[torch.Tensor, int]:
    """signals[] entry -> ([T,3,H,W] in [-1,1], pre-cap frame count).

    Normalization always covers the full recording (Stage-1 contract); the
    even-spaced cap (first+last kept) only ever binds on long haptic tails."""
    signal, family = load_signal(sig_entry)
    frames = family_frames(signal, family, CELL)
    total = frames.shape[0]
    if total > max_frames:
        picks = sorted({round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)})
        frames = frames[picks]
    return frames, total


def strip_array(frames: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] float [-1,1] -> [T*H, W] uint8 (channels are identical)."""
    t, _, h, w = frames.shape
    u8 = ((frames[:, 0] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return u8.reshape(t * h, w).numpy()


def _render_one(task: tuple) -> tuple[str, str, int, int, str]:
    """(stem, sig_entry, out_dir, max_frames) -> (stem, filename, kept, total, error)."""
    stem, sig_entry, out_dir, max_frames = task
    existing = glob(os.path.join(out_dir, f"{stem}_stack*.png"))
    if existing:
        name = os.path.basename(existing[0])
        return stem, name, int(name[len(stem):].removeprefix("_stack").removesuffix(".png")), -1, ""
    try:
        frames, total = render_frames(sig_entry, max_frames)
    except ValueError as exc:  # non-finite signal or sub-floor frame
        return stem, "", 0, 0, str(exc)
    from PIL import Image

    name = f"{stem}_stack{frames.shape[0]}.png"
    tmp = os.path.join(out_dir, name + ".tmp")
    Image.fromarray(strip_array(frames), "L").save(tmp)
    os.replace(tmp, os.path.join(out_dir, name))
    return stem, name, frames.shape[0], total, ""


def image_stem(row: dict) -> str:
    """The row's plot-PNG md5 stem: the per-signal uid shared with ts_images."""
    images = row.get("images") or []
    assert len(images) == 1, f"expected one plot image, got {len(images)}"
    return Path(images[0]["image"]).stem


def to_video_row(row: dict, strip_path: str) -> dict:
    """Swap the plot image for the strip; everything else stays byte-identical."""
    prompt = [dict(m) for m in row["prompt"]]
    assert sum(m["content"].count("<image>") for m in prompt) == 1
    for m in prompt:
        m["content"] = m["content"].replace("<image>", "<video>")
    return {
        **row,
        "prompt": prompt,
        "images": [],
        "videos": [{"video": strip_path, "min_frames": None, "max_frames": None}],
    }


def load_join(data_root: str, raw_root: str, family: str, split: str) -> tuple[list[dict], list[dict], object]:
    import pyarrow.parquet as pq

    table = pq.read_table(f"{data_root}/{family}_{split}.parquet")
    rows = table.to_pylist()
    raw = pq.read_table(f"{raw_root}/{family}_{split}.parquet", columns=["reward_model", "signals"]).to_pylist()
    assert len(rows) == len(raw), f"{family}_{split}: {len(rows)} grpo vs {len(raw)} raw rows"
    for i, (g, r) in enumerate(zip(rows, raw)):
        assert g["reward_model"]["ground_truth"] == r["reward_model"]["ground_truth"], f"{family}_{split}[{i}]: join mismatch"
    return rows, raw, table.schema


def build(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for family in args.families:
        out_dir = os.path.join(args.stage_root, family)
        os.makedirs(out_dir, exist_ok=True)
        for split in args.splits:
            rows, raw, schema = load_join(args.data_root, args.raw_root, family, split)
            tasks, stems = {}, []
            for g, r in zip(rows, raw):
                stem = image_stem(g)
                stems.append(stem)
                tasks.setdefault(stem, (stem, r["signals"][0], out_dir, args.max_frames))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = {res[0]: res for res in pool.map(_render_one, tasks.values(), chunksize=16)}

            out_rows, dropped, capped, kept_counts = [], [], 0, []
            for row, stem in zip(rows, stems):
                _, name, kept, total, error = results[stem]
                if error:
                    dropped.append((stem, error))
                    continue
                capped += total > kept  # skip-existing reports total=-1: never counted
                kept_counts.append(kept)
                out_rows.append(to_video_row(row, os.path.join(out_dir, name)))
            # GAP before the first TS_NATIVE run: run_qwen35_grpo.sh expects RL-half
            # train files (split_grpo/rl/smellnet_train_base_tsnative.parquet), but
            # this writes UNSPLIT root-level files and load_join row-index-asserts
            # against the unsplit trainedve_raw corpora -- needs a base-filter/output
            # suffix and a split-aware join (or a documented rename+re-split step).
            out = f"{args.data_root}/{family}_{split}_tsnative.parquet"
            pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), out)
            counts = sorted(kept_counts)
            print(
                f"{family}_{split}: wrote {len(out_rows)} (dropped {len(dropped)}, capped {capped}) "
                f"frames min/med/max {counts[0]}/{counts[len(counts) // 2]}/{counts[-1]} -> {out}",
                flush=True,
            )
            for stem, error in dropped[:5]:
                print(f"  DROPPED {stem}: {error}")


def probe(args) -> None:
    """Render 3 rows/family, save inspectable PNGs, and measure that the
    student chain (strip -> MIRLDataset fetch -> HF video processor, default
    flags) reproduces the Stage-1 tensor feed."""
    from PIL import Image
    from transformers import AutoProcessor

    from mirl_ext.data.dataset import MIRLDataset

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    for family in args.families:
        rows, raw, _ = load_join(args.data_root, args.raw_root, family, "train")
        # First row of every data_source (covers smellnet base+mixture), then fill to 3.
        by_source: dict[str, int] = {}
        for i, g in enumerate(rows):
            by_source.setdefault(g["data_source"], i)
        picks = sorted(by_source.values())
        picks += [i for i in range(len(rows)) if i not in picks][: max(0, 3 - len(picks))]
        picks = sorted(picks)[:3]
        out_dir = Path(args.probe) / family
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in picks:
            stem = image_stem(rows[i])
            frames, total = render_frames(raw[i]["signals"][0], args.max_frames)
            t = frames.shape[0]
            strip_path = out_dir / f"{stem}_stack{t}.png"
            Image.fromarray(strip_array(frames), "L").save(strip_path)
            for k in (0, t // 2, t - 1):  # 8x nearest so 32 px tiles are inspectable
                frame = Image.fromarray(strip_array(frames[k : k + 1]), "L")
                frame.resize((frame.width * 8, frame.height * 8), Image.NEAREST).save(out_dir / f"{stem}_f{k:04d}_x8.png")

            messages = [{"role": "user", "content": [{"type": "video", "video": str(strip_path)}]}]
            _, videos, _ = MIRLDataset._process_multi_modal_info(messages, image_patch_size=CELL // 2, config=None)
            video, meta = videos[0]
            student = processor.video_processor(
                videos=[video], video_metadata=[meta], do_sample_frames=False, return_tensors="pt"
            )
            reference = processor.video_processor(
                [frames],
                do_convert_rgb=False,
                do_sample_frames=False,
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                return_tensors="pt",
            )
            grid_s = student["video_grid_thw"].tolist()
            grid_r = reference["video_grid_thw"].tolist()
            diff = (student["pixel_values_videos"] - reference["pixel_values_videos"]).abs().max().item() if grid_s == grid_r else float("nan")
            tokens = sum(gt * gh * gw for gt, gh, gw in grid_s) // 4
            status = "OK" if grid_s == grid_r and diff < 0.009 else "MISMATCH"
            print(
                f"{family}[{i}] {rows[i]['data_source']}: frames {t}/{total} grid {grid_s} "
                f"tokens {tokens} |student-ref|max {diff:.5f} [{status}]",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--splits", nargs="*", default=["train", "valid"], choices=["train", "valid"])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--raw-root", default=f"{DATA_ROOT}/trainedve_raw")
    ap.add_argument("--stage-root", default=f"{SCRATCH_ROOT}/data/ts_native")
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--probe", default=None, help="probe output dir; renders 3 rows/family instead of building")
    ap.add_argument("--model-path", default=os.environ.get("MIRL_QWEN35_PATH", ""), help="Qwen3.5 snapshot (probe only)")
    args = ap.parse_args()
    probe(args) if args.probe else build(args)


if __name__ == "__main__":
    main()
