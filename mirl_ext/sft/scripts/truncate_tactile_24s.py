"""Create 24 s truncated copies of long tactile recordings and repoint parquets.

The tactile long tail (>24 s, ~12% of recordings) is repeated squeeze cycles;
the first 24 s carry the signal (decision 2026-09-02). This script:
  1. finds every video referenced by the tactile parquets that runs > TRUNC_S,
  2. writes a stream-copied 24 s mp4 to  <video_dir>/../truncated24/<same name>,
  3. writes a sliced .pt (tactile maps + force stats, first 720 frames) to
     <pt_dir>/../truncated24_pt/<same name>,
  4. rewrites the tactile train/valid parquets to reference the truncated mp4s
     (originals backed up next to each parquet as *.pre_trunc24.parquet).

Full recordings stay on disk untouched; identical basenames keep every
stem-join (splits, annotations, staging) valid. Recordings <= 24 s are left
alone everywhere.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import DATA_ROOT  # noqa: E402

TRUNC_S = 24.0
TS_FRAMES = int(TRUNC_S * 30)
PARQUET_PATHS = (
    "tactile_train.parquet",
    "tactile_valid.parquet",
    "tactile_valid_fast.parquet",
    "tactile_valid_fast_closed.parquet",
    "split_grpo/rl/tactile_train_closed.parquet",
    "split_grpo/sft/tactile_train.parquet",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from torchcodec.decoders import VideoDecoder

    videos: set[str] = set()
    tables = {}
    for relative_path in PARQUET_PATHS:
        p = args.data_root / relative_path
        if not p.is_file():
            print(f"[skip] missing {p}")
            continue
        tables[p] = pq.read_table(p)
        for vids in tables[p].column("videos").to_pylist():
            for v in vids or []:
                raw = v.get("video") if isinstance(v, dict) else v
                if raw:
                    videos.add(str(raw))
    print(f"{len(videos)} unique videos across {len(tables)} parquets")

    remap: dict[str, str] = {}
    n_long = n_pt = 0
    for src in sorted(videos):
        sp = Path(src)
        if not sp.is_file():
            continue
        try:
            meta = VideoDecoder(src).metadata
            duration = meta.num_frames / meta.average_fps
        except Exception as e:
            print(f"  ERR probe {sp.name}: {e}")
            continue
        if duration <= TRUNC_S + 0.5:
            continue
        n_long += 1
        out_dir = sp.parent.parent / "truncated24" / sp.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / sp.name
        if not dst.is_file():
            r = subprocess.run(
                [args.ffmpeg, "-y", "-loglevel", "error", "-i", src, "-t", str(TRUNC_S), "-c", "copy", str(dst)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 or not dst.is_file():
                print(f"  ERR ffmpeg {sp.name}: {r.stderr.strip()[:120]}")
                continue
        remap[src] = str(dst)

    # truncated .pt twins for the same recordings (used by v2 staging / future ts
    # arms). Paths come from the signals[] columns -- the same source the v2
    # stager resolves -- never from directory guessing.
    import glob as _glob

    import torch

    from mirl_ext.sft.scripts.stage_tactile_v2 import stem_to_pt

    pt_map = stem_to_pt(sorted(_glob.glob(str(args.data_root / "trainedve_raw/*.parquet"))))
    for stem, (pt_path, _key) in sorted(pt_map.items()):
        pt = Path(pt_path)
        if not pt.is_file() or "truncated24_pt" in str(pt):
            continue
        dst = pt.parent.parent / "truncated24_pt" / pt.parent.name / pt.name
        if dst.is_file():
            n_pt += 1
            continue
        o = torch.load(pt, map_location="cpu", weights_only=False)
        tac = o.get("tactile")
        if not isinstance(tac, dict) or not tac:
            continue
        if max(v.shape[0] for v in tac.values()) <= TS_FRAMES:
            continue  # already <= 24 s: no twin needed, consumers use the original
        o["tactile"] = {k: v[:TS_FRAMES] for k, v in tac.items()}
        if o.get("hand_force_stats") is not None:
            o["hand_force_stats"] = o["hand_force_stats"][:TS_FRAMES]
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(o, dst)
        n_pt += 1

    print(f"long videos truncated: {n_long} (remapped {len(remap)}); pt twins: {n_pt}")

    for p, table in tables.items():
        rows = table.to_pylist()
        changed = 0
        for row in rows:
            for v in row.get("videos") or []:
                if isinstance(v, dict) and v.get("video") in remap:
                    v["video"] = remap[v["video"]]
                    changed += 1
        if not changed:
            print(f"[unchanged] {p.name}")
            continue
        backup = p.with_name(p.stem + ".pre_trunc24.parquet")
        if not backup.is_file():
            shutil.copy(p, backup)
        pq.write_table(pa.Table.from_pylist(rows), p)
        print(f"[rewritten] {p} rows_touched={changed} (backup: {backup.name})")


if __name__ == "__main__":
    main()
