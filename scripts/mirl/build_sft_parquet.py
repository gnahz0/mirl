# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Join validated traces back to their rows and emit veRL SFT parquet.

Output schema matches ``verl/utils/dataset/multiturn_sft_dataset.py``:

    messages : [{"role": "user", "content": <prompt>},
                {"role": "assistant", "content": <trace>}]
    images   : the row's original ``images`` list, untouched
    videos   : likewise

The user turn is the SFT half's prompt **verbatim**, `<image>` placeholder and all.
That is deliberate: GRPO will present the identical string, and a train/serve template
mismatch is invisible in the loss but shows up as a silent capability gap. Copying the
prompt rather than rebuilding it also means the image placeholder count keeps matching
``len(images)``, which the dataset asserts.

Rows whose trace is missing are simply absent -- no placeholder responses, since an
empty assistant turn would train the model to emit nothing.

    python scripts/mirl/build_sft_parquet.py \\
        --traces data/sft/ts_traces_grounded.jsonl \\
        --out /work/.../data/split_grpo/sft_parquet
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


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


def sft_messages(row: dict) -> list[dict]:
    """Prompt turns reshaped for ``MultiTurnSFTDataset``: leading system merged into user.

    That class tokenizes each message IN ISOLATION (``messages=[message]``) so it can
    build a per-turn loss mask, and Qwen3.5's chat template refuses to render a
    system-only list -- it raises "No user query found in messages". So an explicit
    system turn cannot survive this path, however correct it looks in the parquet.

    KNOWN CONSEQUENCE, do not lose this: GRPO's rl_dataset renders the full list, so at
    RL time our framing occupies a real ``system`` turn. Here it becomes the head of the
    ``user`` turn, under the template's default "You are a helpful assistant." system
    prompt. The model sees the same words in a different structural slot. That is a
    genuine train/serve rendering difference; it affects smellnet/climb/tactile (which
    have a system turn) and not ecg/haptic_ts (which do not). Fixing it properly means
    either teaching the SFT path to batch-render turns, or dropping the system turn from
    the RL indexes so both sides agree.
    """
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
    ap.add_argument("--split-root", default="/work/mit/ppliang_mit/alecz/data/split_grpo")
    ap.add_argument("--out", default="/work/mit/ppliang_mit/alecz/data/split_grpo/sft_parquet")
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
            # Same guard as the verifier: a shifted index would pair a trace with a
            # different example's image and label, and nothing downstream would notice.
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
