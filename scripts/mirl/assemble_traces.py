# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Merge per-family trace files into one corpus, with provenance and sanity gates.

Generation ran per family with different gallery configurations, so the outputs live
in separate files and one family (smellnet) was regenerated after its gallery grew
from 1 to 3 examples per class. Merging by hand risks two silent mistakes: keeping
the superseded traces, and letting two files disagree about the same row.

Keyed on ``(family, row_index)`` rather than ``uid``: uid is "<family>#<row_index>"
and row_index restarts per split half, so uid alone is ambiguous the moment a file
drawn from a different half enters the mix -- the bug that once excluded 94 of 121
smellnet queries.

Later sources win, so list files in increasing order of preference.

    python scripts/mirl/assemble_traces.py \\
        --inputs data/sft/ts_traces_v2.jsonl \\
                 data/sft/ts_traces_smellnet_v3.jsonl \\
                 data/sft/ts_traces_haptic_v3.jsonl \\
        --drop-family-from smellnet_train=data/sft/ts_traces_v2.jsonl \\
        --out data/sft/ts_traces_final.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

_GALLERY_LEAK = re.compile(r"\b(galler(y|ies)|references?|examples)\b", re.IGNORECASE)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument(
        "--drop-family-from",
        nargs="*",
        default=[],
        metavar="FAMILY=FILE",
        help="discard FAMILY's rows coming from FILE (used to drop traces superseded "
        "by a later regeneration under a different gallery configuration)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drops: set[tuple[str, str]] = set()
    for spec in args.drop_family_from:
        fam, _, f = spec.partition("=")
        drops.add((fam, f))

    merged: dict[tuple, dict] = {}
    dropped = 0
    for path_s in args.inputs:
        path = Path(path_s)
        if not path.is_file():
            print(f"[skip] {path} not found")
            continue
        rows = load(path)
        kept = 0
        for r in rows:
            if (r["family"], path_s) in drops:
                dropped += 1
                continue
            merged[(r["family"], r["row_index"])] = {**r, "_source": path.name}
            kept += 1
        print(f"  {path.name:34s} rows={len(rows):5d} kept={kept:5d}")

    out = list(merged.values())
    by_fam = collections.Counter(r["family"] for r in out)
    print(f"\nmerged {len(out)} traces  {dict(by_fam)}   (superseded dropped: {dropped})")

    # Gates. Each is something that silently poisons the corpus if it slips through.
    problems = []
    leaks = [r for r in out if _GALLERY_LEAK.search(r["response"])]
    if leaks:
        problems.append(f"{len(leaks)} traces cite a gallery the SFT prompt will not contain")
    bad_fmt = [
        r for r in out
        if "</think>" not in r["response"] or "\\boxed{" not in r["response"]
    ]
    if bad_fmt:
        problems.append(f"{len(bad_fmt)} traces are missing </think> or \\boxed{{}}")
    models = collections.Counter(r.get("model") for r in out)
    if len(models) > 1:
        problems.append(f"mixed model provenance: {dict(models)}")
    ungrounded = [r for r in out if not r.get("grounded", True)]
    if ungrounded:
        problems.append(f"{len(ungrounded)} traces were generated WITHOUT their plot")

    print(f"provenance: {dict(models)}")
    print(f"sources   : {dict(collections.Counter(r['_source'] for r in out))}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nall gates passed")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as fh:
        for r in sorted(out, key=lambda x: (x["family"], x["row_index"])):
            fh.write(json.dumps(r) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
