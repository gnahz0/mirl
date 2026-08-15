"""Join accepted traces back to their split rows and emit veRL SFT parquet.

Per accepted uid: re-read the original SFT-half row, assert its ground truth
equals the one the trace was validated against (a shifted index would pair a
trace with the wrong media/label), keep the original prompt and every original
image/video in order, append exactly one assistant turn with the accepted
completion, and carry the original extra_info plus generation provenance.
Exhausted/error records and teacher-only staging never enter the parquet.

    python mirl_ext/sft/build_sft_parquet.py --traces data/sft/traces.jsonl --out .../sft_parquet
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sft_tasks import DATA_ROOT, extra, prompt_messages  # noqa: E402


def sft_messages(row: dict) -> list[dict]:
    """Leading system turn merged into the user turn: MultiTurnSFTDataset
    tokenizes each message in isolation and Qwen3.5's template rejects a
    system-only list (tests/test_sft_pipeline.py::test_system_turn_workaround
    documents this). KNOWN train/serve difference: GRPO renders a true system
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


def accepted_traces(paths: list[Path]) -> dict[str, dict]:
    """uid -> last accepted record. Resume files may hold several records per
    uid (an error line later superseded); the LAST line per uid wins, and only
    accepted ones build rows. Records without a status are legacy accepted.
    Unparseable lines (kill-truncated tails) are skipped like read_status does."""
    last: dict[str, dict] = {}
    skipped = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    last[rec["uid"]] = rec
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
    if skipped:
        print(f"[warn] skipped {skipped} unparseable trace line(s)")
    return {
        uid: rec for uid, rec in last.items() if rec.get("status", "accepted") == "accepted"
    }


def build_record(row: dict, trace: dict) -> dict:
    gt = (row.get("reward_model") or {}).get("ground_truth")
    assert gt == trace["ground_truth"], f"{trace['uid']}: join mismatch, refusing to write"
    provenance = {
        "uid": trace["uid"],
        "family": trace["family"],
        "ground_truth": gt,
        "teacher_model": trace.get("model"),
        "mode": trace.get("mode"),
        "prompt_version": trace.get("prompt_version"),
        "accepted_attempt": trace.get("accepted_attempt"),
        "staging_version": trace.get("staging_version"),
    }
    return {
        "data_source": row.get("data_source"),
        "messages": sft_messages(row) + [{"role": "assistant", "content": trace["response"]}],
        "images": row.get("images") or [],
        "videos": row.get("videos") or [],
        "extra_info": json.dumps({**extra(row), **provenance}),
    }


def check_record(record: dict, uid: str) -> None:
    """Placeholder counts must match media counts or the SFT dataset asserts."""
    text = "".join(m["content"] for m in record["messages"])
    n_img, n_vid = text.count("<image>"), text.count("<video>")
    assert n_img == len(record["images"]), f"{uid}: {n_img} <image> vs {len(record['images'])}"
    assert n_vid == len(record["videos"]), f"{uid}: {n_vid} <video> vs {len(record['videos'])}"
    assert record["messages"][-1]["role"] == "assistant" and record["messages"][-1]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traces", nargs="+", required=True)
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

    traces = accepted_traces([Path(p) for p in args.traces])
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for t in traces.values():
        by_family[t["family"]].append(t)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    total = 0

    for family, items in sorted(by_family.items()):
        rows = pq.read_table(Path(args.split_root) / "sft" / f"{family}.parquet").to_pylist()
        records = []
        for t in items:
            record = build_record(rows[t["row_index"]], t)
            check_record(record, t["uid"])
            records.append(record)
        if args.single_file:
            combined.extend(records)
        total += len(records)
        n_img = sum(1 for r in records if r["images"])
        n_vid = sum(1 for r in records if r["videos"])
        print(
            f"{family:22s} accepted={len(items):6d} written={len(records):6d} "
            f"with_images={n_img} with_videos={n_vid}"
        )
        if not args.single_file:
            pq.write_table(pa.Table.from_pylist(records), out_root / f"{family}_sft.parquet")

    if args.single_file:
        pq.write_table(pa.Table.from_pylist(combined), out_root / "mirl_sft.parquet")
        print(f"\nwrote {total} rows -> {out_root / 'mirl_sft.parquet'}")
    else:
        print(f"\nwrote {total} rows across {len(by_family)} files -> {out_root}")
    print("Smoke-test with a tiny trainer run (data.train_max_samples=16) before training.")


if __name__ == "__main__":
    main()
