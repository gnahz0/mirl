"""Join validated traces back to their rows and emit veRL SFT parquet.

The user turn is the SFT half's prompt VERBATIM (placeholder and all): GRPO will
present the identical string, and a train/serve template mismatch is invisible in
the loss. Rows without a trace are simply absent.

    python mirl_ext/sft/build_sft_parquet.py --traces data/sft/traces.jsonl --out .../sft_parquet
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sft_tasks import prompt_messages  # noqa: E402  (single home for the prompt[0] lesson)
from paths import DATA_ROOT  # noqa: E402


def sft_messages(row: dict) -> list[dict]:
    """Leading system turn merged into the user turn: MultiTurnSFTDataset
    tokenizes each message in isolation and Qwen3.5's template rejects a
    system-only list. KNOWN train/serve difference: GRPO renders a true system
    turn; here the same words ride at the head of the user turn."""
    msgs = prompt_messages(row)
    head = [m for m in msgs if m["role"] == "system"]
    rest = [m for m in msgs if m["role"] != "system"]
    if not head:
        return rest or msgs
    if not rest:
        return [{"role": "user", "content": "\n\n".join(m["content"] for m in head)}]
    merged = "\n\n".join([m["content"] for m in head] + [rest[0]["content"]])
    return [{"role": "user", "content": merged}] + rest[1:]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traces", required=True)
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--out", default=f"{DATA_ROOT}/split_grpo/sft_parquet")
    ap.add_argument(
        "--single-file",
        action="store_true",
        help="write one combined parquet instead of one per family",
    )
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    traces = [json.loads(l) for l in Path(args.traces).read_text().splitlines() if l.strip()]
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for t in traces:
        by_family[t["family"]].append(t)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    total = 0

    for family, items in sorted(by_family.items()):
        rows = pq.read_table(Path(args.split_root) / "sft" / f"{family}.parquet").to_pylist()
        records = []
        for t in items:
            row = rows[t["row_index"]]
            gt = (row.get("reward_model") or {}).get("ground_truth")
            # A shifted index would pair a trace with the wrong image/label.
            assert gt == t["ground_truth"], f"{t['uid']}: join mismatch, refusing to write"
            records.append(
                {
                    "data_source": row.get("data_source"),
                    "messages": sft_messages(row)
                    + [{"role": "assistant", "content": t["response"]}],
                    "images": row.get("images") or [],
                    "videos": row.get("videos") or [],
                    "extra_info": json.dumps(
                        {
                            "uid": t["uid"],
                            "family": family,
                            "ground_truth": gt,
                            "gen_model": t.get("model"),
                            "grounded": bool(t.get("grounded")),
                        }
                    ),
                }
            )
        if args.single_file:
            combined.extend(records)
        total += len(records)
        n_img = sum(1 for r in records if r["images"])
        print(f"{family:22s} traces={len(items):6d} written={len(records):6d} with_images={n_img}")
        if not args.single_file:
            pq.write_table(pa.Table.from_pylist(records), out_root / f"{family}_sft.parquet")

    if args.single_file:
        pq.write_table(pa.Table.from_pylist(combined), out_root / "mirl_sft.parquet")
        print(f"\nwrote {total} rows -> {out_root / 'mirl_sft.parquet'}")
    else:
        print(f"\nwrote {total} rows across {len(by_family)} files -> {out_root}")
    print("Point the SFT launcher's train_files at these.")


if __name__ == "__main__":
    main()
