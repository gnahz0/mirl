# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Prove generated SFT traces would actually earn GRPO reward on our own rows.

The generator validates against ``format_reward`` + an exact boxed-answer match.
That is necessary but NOT sufficient: GRPO scores through
``mirl_ext.rewards.combined.compute_score``, which dispatches per ``data_source``
and applies task-specific parsing on top (ECG snaps the answer to its 7-category
vocabulary, tactile/human-behaviour use set overlap, etc.). A trace can satisfy the
format regex and still score poorly there.

So this replays each trace through the REAL reward the RL run will use, joined back
to the REAL row in the SFT parquet, and reports the score distribution per family.
Anything below 1.0 is shown, because that is a target the student would be trained
to produce and then penalized for at RL time.

    # on the cluster (needs the split parquets)
    python scripts/mirl/verify_sft_traces.py --traces data/sft/traces.jsonl --show 2
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.rewards.combined import compute_score  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traces", required=True)
    ap.add_argument("--split-root", default="/work/mit/ppliang_mit/alecz/data/split_grpo")
    ap.add_argument("--show", type=int, default=1, help="worked examples to print per family")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    traces = [json.loads(l) for l in Path(args.traces).read_text().splitlines() if l.strip()]
    if args.limit:
        traces = traces[: args.limit]
    by_family = collections.defaultdict(list)
    for t in traces:
        by_family[t["family"]].append(t)

    print(f"verifying {len(traces)} traces against mirl_ext.rewards.combined\n")
    all_scores, shown_total = [], 0
    for family, items in sorted(by_family.items()):
        rows = pq.read_table(Path(args.split_root) / "sft" / f"{family}.parquet").to_pylist()
        scores, bad = [], []
        for t in items:
            row = rows[t["row_index"]]
            gt_row = (row.get("reward_model") or {}).get("ground_truth")
            # The join must be exact: a shifted row_index would silently score the
            # trace against a DIFFERENT example's label and still look plausible.
            assert gt_row == t["ground_truth"], (
                f"{t['uid']}: parquet/trace ground-truth mismatch -- join is wrong"
            )
            res = compute_score(
                data_source=row.get("data_source"),
                solution_str=t["response"],
                ground_truth=gt_row,
            )
            scores.append(res["score"])
            if res["score"] < 1.0:
                bad.append((t, row, res))
        all_scores.extend(scores)
        print(
            f"{family:26s} n={len(scores):5d}  mean={statistics.mean(scores):.4f}  "
            f"min={min(scores):.4f}  perfect={sum(1 for s in scores if s >= 0.999)}/{len(scores)}"
        )
        if bad:
            t, row, res = bad[0]
            print(f"    LOW: {t['uid']} score={res['score']:.3f} acc={res.get('acc')} "
                  f"fmt={res.get('format', res.get('fmt'))}")
            print(f"         gt={t['ground_truth'][:90]!r}")

        for t in items[: args.show]:
            if shown_total >= args.show * 2:
                break
            row = rows[t["row_index"]]
            res = compute_score(
                data_source=row.get("data_source"),
                solution_str=t["response"],
                ground_truth=t["ground_truth"],
            )
            print("\n" + "=" * 78)
            print(f"WORKED EXAMPLE  uid={t['uid']}  family={family}  "
                  f"data_source={row.get('data_source')}")
            print("=" * 78)
            print(f"[1] REAL ROW media: {(row.get('signals') or row.get('images') or row.get('videos') or [{}])[0]}")
            print(f"\n[2] PROMPT (from our parquet, verbatim):\n{prompt_text(row)[:700]}")
            print(f"\n[3] GROUND TRUTH (from our parquet):\n{t['ground_truth'][:400]}")
            print(f"\n[4] GENERATED TRACE ({t['model']}, attempt {t['attempts']}):\n{t['response'][:900]}")
            print(f"\n[5] REAL GRPO REWARD (mirl_ext.rewards.combined.compute_score):")
            for k, v in res.items():
                print(f"      {k:12s} {v}")
            shown_total += 1
        print()

    print("=" * 78)
    print(
        f"OVERALL n={len(all_scores)} mean={statistics.mean(all_scores):.4f} "
        f"perfect={sum(1 for s in all_scores if s >= 0.999)}/{len(all_scores)}"
    )


if __name__ == "__main__":
    main()
