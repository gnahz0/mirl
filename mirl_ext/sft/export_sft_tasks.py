"""Export a split half to a compact task JSONL for trace generation.

Runs on the cluster (where the parquets live); the JSONL is small enough to copy
to a laptop, which has the outbound internet the compute nodes lack. Each task
carries the FULL original prompt (all system+user turns flattened), every media
reference in original order, its ground truth (for laptop-side validation only
-- the generator strips it before building requests), and derived label
metadata: extracted in-prompt choices, an answer_style (closed = exact-match
gradable, open = free text), and a per-source label_space when the prompt lists
no options. SmellNet exports the 50-class single-substance task only; mixture
and GC-MS rows are excluded and asserted absent. Sampling is stratified by
(data_source, label) so rare classes survive the per-family cap.

    python mirl_ext/sft/export_sft_tasks.py --limit-per-family 2500 --out .../sft_tasks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
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

# Raw-signal families whose rows carry a rendered plot PNG readable by GPT vision.
TS_FAMILIES = ["smellnet_train", "ecg_train", "haptic_ts_train"]

_CHOICE_HEAD_RE = re.compile(
    r"(answer with[^:\n]{0,80}following\s*:?|(?<![\w])options\s*:)", re.IGNORECASE
)


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


def extract_choices(prompt: str) -> list[str]:
    """Options the prompt itself enumerates (newline list, lettered list, or a
    single comma-separated line), or [] when it lists none."""
    m = _CHOICE_HEAD_RE.search(prompt)
    if not m:
        return []
    block = prompt[m.end() :].split("\n\n", 1)[0].strip()
    if "\n" in block:
        return [line.strip() for line in block.splitlines() if line.strip()]
    return [c.strip() for c in block.split(",") if c.strip()]


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


def source_styles(rows: list[dict]) -> dict[str, dict]:
    """Per data_source: answer_style + label_space, derived from the whole family.

    closed = the prompt enumerates options, or the source's distinct ground
    truths form a small label-like set; open = free text (no exact-match gate).
    label_space is attached only when the prompt lists no options itself.
    """
    gts: dict[str, set] = collections.defaultdict(set)
    has_choices: dict[str, bool] = {}
    for row in rows:
        src = str(row.get("data_source"))
        gt = (row.get("reward_model") or {}).get("ground_truth")
        if gt:
            gts[src].add(str(gt))
        if src not in has_choices:
            has_choices[src] = bool(extract_choices(prompt_text(row)))
    styles = {}
    for src, labels in gts.items():
        small = len(labels) <= 60 and max(len(g) for g in labels) <= 60
        closed = has_choices[src] or small
        styles[src] = {
            "answer_style": "closed" if closed else "open",
            "label_space": sorted(labels) if (closed and small and not has_choices[src]) else None,
        }
    return styles


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[int]:
    """~n row positions, water-filled over (data_source, label) so rare classes survive."""
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
        help="which split half to export. v1 answer-blind generation uses 'sft' "
        "only; 'rl' exists for the (currently unused) episode generator's "
        "support pool and never feeds trace generation.",
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

    half_root = Path(args.split_root) / args.half
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = (TS_FAMILIES if args.ts_only else None) or args.families or FAMILIES
    total = 0
    with out_path.open("w") as fh:
        for family in wanted:
            src = half_root / f"{family}.parquet"
            if not src.exists():
                print(f"[skip] {src} not found")
                continue
            rows = pq.read_table(src).to_pylist()
            styles = source_styles(rows)
            # Keep original parquet positions: uid must join back to the row.
            eligible = [
                (i, row)
                for i, row in enumerate(rows)
                if keep_row(family, row.get("data_source"))
                and str((row.get("reward_model") or {}).get("ground_truth") or "").strip()
            ]
            picked = stratified_sample([r for _, r in eligible], args.limit_per_family, args.seed)
            exported_sources: collections.Counter = collections.Counter()
            labels = set()
            n_media = n_open = 0
            for pos in picked:
                i, row = eligible[pos]
                data_source = row.get("data_source")
                gt = (row.get("reward_model") or {}).get("ground_truth")
                ei = extra(row)
                images, video_path, max_frames = media_refs(row)
                style = styles.get(str(data_source), {"answer_style": "open", "label_space": None})
                text = prompt_text(row)
                task = {
                    # build_sft_parquet joins completions back on uid.
                    "uid": f"{family}#{i}",
                    "family": family,
                    "row_index": i,
                    "source_index": ei.get("index"),
                    "data_source": data_source,
                    "question_type": ei.get("question_type"),
                    "prompt": text,
                    "ground_truth": gt,
                    "image_paths": images,
                    "video_path": video_path,
                    "max_frames": max_frames,
                    "answer_style": style["answer_style"],
                }
                if style["label_space"]:
                    task["label_space"] = style["label_space"]
                fh.write(json.dumps(task) + "\n")
                exported_sources[str(data_source)] += 1
                labels.add(gt)
                n_media += bool(images or video_path)
                n_open += style["answer_style"] == "open"
                total += 1
            if family == "smellnet_train":
                bad = [s for s in exported_sources if "mixture" in s or "gc" in s.lower()]
                assert not bad, f"smellnet export leaked excluded sources: {bad}"
                assert set(exported_sources) <= {"smellnet_base"}, dict(exported_sources)
            print(
                f"{family:22s} half_rows={len(rows):6d} eligible={len(eligible):6d} "
                f"exported={len(picked):6d} labels={len(labels):5d} "
                f"with_media={n_media} open={n_open}"
            )

    print(f"\nwrote {total} tasks -> {out_path}")
    print("Copy to the machine with internet access, stage media, then run gen_sft_targets.py.")


if __name__ == "__main__":
    main()
