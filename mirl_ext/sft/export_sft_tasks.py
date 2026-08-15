"""Export a split half to a compact task JSONL for trace generation.

Runs on the cluster (where the parquets live); the JSONL is small enough to copy
to a laptop, which has the outbound internet the compute nodes lack. Sampling is
stratified by (data_source, label) so rare classes survive the per-family cap.

    python mirl_ext/sft/export_sft_tasks.py --limit-per-family 2500 --out .../sft_tasks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
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

# Raw-signal families whose rows carry a rendered plot PNG GPT-vision can read.
TS_FAMILIES = ["smellnet_train", "ecg_train", "haptic_ts_train"]


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


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[int]:
    """~n row indices, water-filled over (data_source, label) so rare classes survive."""
    buckets: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r.get("data_source"), (r.get("reward_model") or {}).get("ground_truth"))].append(i)
    if n >= len(rows):
        return list(range(len(rows)))

    rng = random.Random(seed)
    for idxs in buckets.values():
        rng.shuffle(idxs)

    chosen: list[int] = []
    remaining = sorted(buckets, key=lambda k: (len(buckets[k]), str(k)))
    budget = n
    for pos, key in enumerate(remaining):
        quota = max(1, budget // max(1, len(remaining) - pos))
        take = min(quota, len(buckets[key]))
        chosen.extend(buckets[key][:take])
        budget -= take
        if budget <= 0:
            break
    return sorted(chosen[:n])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--out", default=f"{DATA_ROOT}/split/sft_tasks.jsonl")
    ap.add_argument("--limit-per-family", type=int, default=2500)
    ap.add_argument(
        "--half",
        default="sft",
        choices=["sft", "rl"],
        help="which split half to export. 'rl' exports the labelled SUPPORT POOL "
        "consumed by gen_sft_episodes --support-tasks: support plots never enter the "
        "SFT parquet (only generated traces do), so RL-half plots never reach the "
        "student -- they only inform how SFT traces are worded. This frees every "
        "SFT-half row to be a query, which is what makes >1 example per class "
        "affordable (base has just 2-3 rows per class per half).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--families", nargs="*", default=None, help="subset of families (default: all)"
    )
    ap.add_argument(
        "--ts-only", action="store_true", help=f"shorthand for --families {' '.join(TS_FAMILIES)}"
    )
    args = ap.parse_args()

    import pyarrow.parquet as pq

    sft_root = Path(args.split_root) / args.half
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = (TS_FAMILIES if args.ts_only else None) or args.families or FAMILIES
    total = 0
    with out_path.open("w") as fh:
        for family in wanted:
            src = sft_root / f"{family}.parquet"
            if not src.exists():
                print(f"[skip] {src} not found")
                continue
            rows = pq.read_table(src).to_pylist()
            picked = stratified_sample(rows, args.limit_per_family, args.seed)
            labels = set()
            n_with_image = 0
            for i in picked:
                row = rows[i]
                gt = (row.get("reward_model") or {}).get("ground_truth")
                if gt is None or str(gt).strip() == "":
                    continue  # no target to rationalize toward; nothing to learn
                labels.add(gt)
                ei = extra(row)
                images = row.get("images") or []
                image_path = ""
                if images:
                    first = images[0]
                    image_path = (first.get("image") or "") if isinstance(first, dict) else str(first)
                videos = row.get("videos") or []
                video_path = ""
                if videos:
                    first = videos[0]
                    video_path = (first.get("video") or "") if isinstance(first, dict) else str(first)
                if image_path or video_path:
                    n_with_image += 1
                fh.write(
                    json.dumps(
                        {
                            # build_sft_parquet joins completions back on uid.
                            "uid": f"{family}#{i}",
                            "family": family,
                            "row_index": i,
                            "data_source": row.get("data_source"),
                            "question_type": ei.get("question_type"),
                            "prompt": prompt_text(row),
                            "ground_truth": gt,
                            "image_path": image_path,
                            "video_path": video_path,
                        }
                    )
                    + "\n"
                )
                total += 1
            print(
                f"{family:22s} sft_rows={len(rows):7d} exported={len(picked):6d} "
                f"distinct_labels={len(labels):5d} with_media={n_with_image}"
            )
            if n_with_image and n_with_image < len(picked):
                # A mix of grounded and ungrounded traces under one family name.
                print(f"    WARNING: only {n_with_image}/{len(picked)} rows carry a plot")

    print(f"\nwrote {total} tasks -> {out_path}")
    print("Copy to the machine with internet access, then run gen_sft_targets.py.")


if __name__ == "__main__":
    main()
