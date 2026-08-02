# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Export the SFT half to a compact text-only task file for CoT generation.

Runs on the cluster (that is where the parquets live) and emits a JSONL small
enough to copy to a laptop, because the generator needs outbound internet and the
compute nodes do not have it.

Only TEXT is exported. GPT never sees the pseudo-image raster -- it is a grayscale
signal encoding with no natural-image semantics, so sending it would cost vision
tokens for no information. The generated `<think>` is therefore an answer-conditioned
*rationalization* (STaR-style), not grounded perception: it teaches the output format
and reasoning shape, and RL grounds it later against the real VE features.

Per-family caps matter. The SFT half is ~140k rows and cold-start needs ~10-20k, so
sampling is stratified by (data_source, label) to keep rare classes represented
rather than letting a head-heavy family dominate the budget.

    python scripts/mirl/export_sft_tasks.py --limit-per-family 2500 \\
        --out /work/.../data/split/sft_tasks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

FAMILIES = [
    "smellnet_train",
    "ecg_train",
    "haptic_ts_train",
    "climb_train",
    "human_behaviour_train",
    "tactile_train",
]

# The three raw-signal families. Their rows carry a rendered plot PNG that GPT-vision
# CAN read (8-lead ECG traces, per-channel e-nose response curves, a taxel heatmap +
# force curve), so their traces can be grounded in the actual signal instead of being
# label-conditioned boilerplate.
TS_FAMILIES = ["smellnet_train", "ecg_train", "haptic_ts_train"]


def prompt_messages(row: dict) -> list[dict]:
    """The row's FULL prompt message list, roles preserved.

    NOT just prompt[0]. smellnet/climb/tactile carry TWO messages -- a system turn
    with the task framing and a user turn holding the real question plus the
    `<image>`/`<video>` placeholder. Reading only the first one silently drops the
    question and the placeholder: it produced smellnet traces answering a question
    that was never asked, and it broke veRL's SFT dataset, which asserts the
    placeholder count matches len(images). ecg/haptic_ts have a single user message,
    which is why they looked fine.
    """
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
    """Sample ~n row indices, spreading the budget evenly over (data_source, label).

    Water-fills: strata smaller than the per-stratum quota give everything they have
    and their unused budget is redistributed, so a cap of 2500 over 132 SmellNet
    classes does not silently return only the head classes.
    """
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
    ap.add_argument("--split-root", default="/work/mit/ppliang_mit/alecz/data/split_grpo")
    ap.add_argument("--out", default="/work/mit/ppliang_mit/alecz/data/split/sft_tasks.jsonl")
    ap.add_argument("--limit-per-family", type=int, default=2500)
    ap.add_argument(
        "--half",
        default="sft",
        choices=["sft", "rl"],
        help="which split half to export. 'rl' is for GALLERY REFERENCES only: the "
        "gallery never enters the SFT parquet (only generated traces do), so RL-half "
        "plots never reach the student -- they only inform how SFT traces are worded. "
        "This frees every SFT-half row to be a query, which is what makes >1 example "
        "per class affordable (base has just 2-3 rows per class per half).",
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
                if image_path:
                    n_with_image += 1
                fh.write(
                    json.dumps(
                        {
                            # uid must survive the round trip: build_sft_parquet joins
                            # completions back on it, so it encodes the exact source row.
                            "uid": f"{family}#{i}",
                            "family": family,
                            "row_index": i,
                            "data_source": row.get("data_source"),
                            "question_type": ei.get("question_type"),
                            "prompt": prompt_text(row),
                            "ground_truth": gt,
                            # Path only. The generator base64-encodes it at call time so
                            # this file stays small enough to copy to a laptop.
                            "image_path": image_path,
                        }
                    )
                    + "\n"
                )
                total += 1
            print(
                f"{family:22s} sft_rows={len(rows):7d} exported={len(picked):6d} "
                f"distinct_labels={len(labels):5d} with_plot={n_with_image}"
            )
            if n_with_image and n_with_image < len(picked):
                # Partial coverage silently produces a MIX of grounded and
                # hallucinated traces under one family name -- worth knowing.
                print(f"    WARNING: only {n_with_image}/{len(picked)} rows carry a plot")

    print(f"\nwrote {total} tasks -> {out_path}")
    print("Copy to the machine with internet access, then run gen_sft_targets.py.")


if __name__ == "__main__":
    main()
