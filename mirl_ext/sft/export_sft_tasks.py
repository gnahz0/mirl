"""Export every eligible row of a split half to a task JSONL for generation.

Runs on the cluster (where the parquets live); the JSONL is small enough to copy
to a laptop, which has the outbound internet the compute nodes lack. Every SFT
row gets a teacher trace, so there is no sampling here -- the 20:80 split is the
sampling. Each task carries the FULL original prompt (all system+user turns
flattened), every media reference in original order, its ground truth (for
laptop-side validation only -- the generator strips it before building
requests), and answer_style: sources in OPEN_SOURCES are free text with no
exact-match gate, everything else is gradable. SmellNet exports the 50-class
single-substance task only; mixture and GC-MS rows are excluded and asserted
absent.

    python mirl_ext/sft/export_sft_tasks.py --out .../sft_tasks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

_CONFIG = Path(__file__).with_name("config.json")


def _config_path(key: str, env: str, fallback: str) -> str:
    """Cluster paths live in config.json (or env overrides), not in code."""
    if os.environ.get(env):
        return os.environ[env].rstrip("/")
    if _CONFIG.is_file():
        cfg = json.loads(_CONFIG.read_text())
        if key in cfg:
            return str(cfg[key]).rstrip("/")
    return fallback


DATA_ROOT = _config_path("cluster_data_root", "MIRL_DATA_ROOT", "data")
SCRATCH_ROOT = _config_path("cluster_scratch_root", "MIRL_SCRATCH_ROOT", "scratch")

FAMILIES = [
    "smellnet_train",
    "ecg_train",
    "haptic_ts_train",
    "climb_train",
    "human_behaviour_train",
    "tactile_train",
]

# Free-text sources (checked once against the data): captions/notes and open QA
# whose answers can't be exact-match validated. Everything else is closed.
OPEN_SOURCES = {
    "haptic_tactile",                                       # haptic_ts descriptions
    "description", "tactile_description", "mat_description",  # tactile captions/notes
    "part_notes", "objA_notes", "objB_notes", "deformation_note",
    "intentqa", "siq2", "mimeqa",                           # free-text video QA
}


def prompt_messages(row: dict) -> list[dict]:
    """The row's FULL prompt message list -- NOT prompt[0]: smellnet/climb/tactile
    carry system + user turns, and dropping the user turn loses the question and
    the <image>/<video> placeholder (this bug shipped once)."""
    p = row.get("prompt")
    if isinstance(p, list):
        out = []
        for m in p:
            if isinstance(m, dict):
                out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                out.append({"role": "user", "content": str(m)})
        return out
    return [{"role": "user", "content": str(p or "")}]


def prompt_text(row: dict) -> str:
    """All message contents joined -- what a text-only API call should receive."""
    return "\n\n".join(m["content"] for m in prompt_messages(row) if m["content"])


def extra(row: dict) -> dict:
    ei = row.get("extra_info")
    if isinstance(ei, str):
        try:
            ei = json.loads(ei)
        except json.JSONDecodeError:
            return {}
    return ei if isinstance(ei, dict) else {}


def keep_row(family: str, data_source: str | None) -> bool:
    """SmellNet keeps only the 50-class single-substance task."""
    if family == "smellnet_train":
        return data_source == "smellnet_base"
    return True


def media_refs(row: dict) -> tuple[list[str], str, int | None]:
    """(image paths in original order, video path, video max_frames)."""
    images = []
    for entry in row.get("images") or []:
        images.append(entry.get("image", "") if isinstance(entry, dict) else str(entry))
    video_path, max_frames = "", None
    videos = row.get("videos") or []
    if videos:
        first = videos[0]
        if isinstance(first, dict):
            video_path = first.get("video") or ""
            max_frames = first.get("max_frames")
        else:
            video_path = str(first)
    return [p for p in images if p], video_path, max_frames


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--out", default=f"{DATA_ROOT}/split/sft_tasks.jsonl")
    ap.add_argument(
        "--half",
        default="sft",
        choices=["sft", "rl"],
        help="split half to export ('rl' only builds the episode-generator support pool)",
    )
    ap.add_argument(
        "--families", nargs="*", default=None, help="subset of families (default: all)"
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    half_root = Path(args.split_root) / args.half
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = args.families or FAMILIES
    total = 0
    with out_path.open("w") as fh:
        for family in wanted:
            src = half_root / f"{family}.parquet"
            if not src.exists():
                print(f"[skip] {src} not found")
                continue
            rows = pq.read_table(src).to_pylist()
            # Keep original parquet positions: uid must join back to the row.
            eligible = [
                (i, row)
                for i, row in enumerate(rows)
                if keep_row(family, row.get("data_source"))
                and str((row.get("reward_model") or {}).get("ground_truth") or "").strip()
            ]
            exported_sources: collections.Counter = collections.Counter()
            labels = set()
            n_media = n_open = 0
            for i, row in eligible:
                data_source = row.get("data_source")
                gt = (row.get("reward_model") or {}).get("ground_truth")
                images, video_path, max_frames = media_refs(row)
                style = "open" if str(data_source) in OPEN_SOURCES else "closed"
                task = {
                    # build_sft_parquet joins completions back on uid.
                    "uid": f"{family}#{i}",
                    "family": family,
                    "row_index": i,
                    "data_source": data_source,
                    "prompt": prompt_text(row),
                    "ground_truth": gt,
                    "image_paths": images,
                    "video_path": video_path,
                    "max_frames": max_frames,
                    "answer_style": style,
                }
                fh.write(json.dumps(task) + "\n")
                exported_sources[str(data_source)] += 1
                labels.add(gt)
                n_media += bool(images or video_path)
                n_open += style == "open"
                total += 1
            if family == "smellnet_train":
                bad = [s for s in exported_sources if "mixture" in s or "gc" in s.lower()]
                assert not bad, f"smellnet export leaked excluded sources: {bad}"
                assert set(exported_sources) <= {"smellnet_base"}, dict(exported_sources)
            print(
                f"{family:22s} half_rows={len(rows):6d} exported={len(eligible):6d} "
                f"labels={len(labels):5d} with_media={n_media} open={n_open}"
            )

    print(f"\nwrote {total} tasks -> {out_path}")
    print("Copy to the machine with internet access, stage media, then run gen_sft_targets.py.")


if __name__ == "__main__":
    main()
