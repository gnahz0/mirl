"""Teacher-performance report over generation record JSONLs.

Everything is derived from the records gen_sft_targets.py writes (one per
attempted task, status accepted/exhausted/error, per-attempt rejection
reasons), so yield IS teacher accuracy. --audit N additionally prints a
stratified sample of accepted traces for the manual grounding read automatic
validation cannot replace.

    python mirl_ext/sft/report_traces.py data/sft/traces.jsonl --audit 6
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path


def load_last_records(paths: list[Path]) -> list[dict]:
    """Last record per uid; unparseable lines (kill-truncated tails) skipped."""
    last: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    last[rec["uid"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return list(last.values())


def attempt_list(rec: dict) -> list[dict]:
    """Per-attempt rejections; legacy records stored an int attempt count."""
    attempts = rec.get("attempts")
    return attempts if isinstance(attempts, list) else []


def accepted_on_first(rec: dict) -> bool:
    first = rec.get("accepted_attempt")
    if first is None and isinstance(rec.get("attempts"), int):  # legacy schema
        first = rec["attempts"]
    return (first or 1) == 1


def think_len(response: str) -> int:
    m = re.search(r"<think>(.*?)</think>", response or "", re.DOTALL)
    return len((m.group(1) if m else "").split())


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "-"


def report(records: list[dict]) -> None:
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)

    print(f"{'family':24s} {'attempted':>9s} {'accepted':>8s} {'pass@1':>7s} "
          f"{'pass@k':>7s} {'exhaust':>7s} {'error':>6s}")
    for fam, recs in sorted(by_family.items()):
        attempted = [r for r in recs if r.get("status") != "error"]
        acc = [r for r in recs if r.get("status", "accepted") == "accepted"]
        p1 = sum(1 for r in acc if accepted_on_first(r))
        print(f"{fam:24s} {len(attempted):9d} {len(acc):8d} {pct(p1, len(attempted)):>7s} "
              f"{pct(len(acc), len(attempted)):>7s} "
              f"{sum(1 for r in recs if r.get('status') == 'exhausted'):7d} "
              f"{sum(1 for r in recs if r.get('status') == 'error'):6d}")

    all_attempts = [a for r in records for a in attempt_list(r)]
    n_completions = len(all_attempts) + sum(
        1 for r in records if r.get("status", "accepted") == "accepted"
    )
    malformed = sum(1 for a in all_attempts if a["reason"] not in ("wrong", "wrong_out_of_space"))
    leaks = sum(1 for a in all_attempts if a["reason"] == "leak")
    print(f"\ncompletions={n_completions} malformed_rate={pct(malformed, n_completions)} "
          f"leak_rejection_rate={pct(leaks, n_completions)}")
    reasons = collections.Counter(a["reason"] for a in all_attempts)
    if reasons:
        print(f"rejection reasons: {dict(reasons.most_common())}")

    lens = [think_len(r.get("response")) for r in records
            if r.get("status", "accepted") == "accepted" and r.get("response")]
    if lens:
        print(f"rationale words: mean={statistics.mean(lens):.0f} "
              f"median={statistics.median(lens):.0f} min={min(lens)} max={max(lens)}")

    for fam, recs in sorted(by_family.items()):
        srcs = collections.defaultdict(lambda: [0, 0])   # data_source -> [accepted, attempted]
        classes = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            if r.get("status") == "error":
                continue
            for table, key in ((srcs, r.get("data_source")), (classes, str(r.get("ground_truth"))[:48])):
                table[key][1] += 1
                table[key][0] += r.get("status", "accepted") == "accepted"
        covered = sum(1 for a, _ in classes.values() if a)
        print(f"\n[{fam}] class coverage: {covered}/{len(classes)} attempted classes have >=1 trace")
        if len(srcs) > 1:
            for src, (a, n) in sorted(srcs.items(), key=lambda kv: -kv[1][1]):
                print(f"  {src:24s} {a:5d}/{n:<5d} {pct(a, n)}")
        worst = sorted(classes.items(), key=lambda kv: (kv[1][0] / max(1, kv[1][1]), -kv[1][1]))[:8]
        for cls, (a, n) in worst:
            print(f"    class {cls:48s} {a:4d}/{n:<4d} {pct(a, n)}")


def audit(records: list[dict], n: int) -> None:
    accepted = [r for r in records if r.get("status", "accepted") == "accepted" and r.get("response")]
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in accepted:
        by_family[r["family"]].append(r)
    print("\n" + "=" * 72 + "\nMANUAL AUDIT SAMPLE (check: query-specific evidence, no invented "
          "measurements,\nno label boilerplate, no plot-formatting talk, sound alternatives)\n" + "=" * 72)
    picked = 0
    while picked < n and any(by_family.values()):
        for fam in sorted(by_family):
            if by_family[fam] and picked < n:
                r = by_family[fam].pop(len(by_family[fam]) // 2)
                picked += 1
                print(f"\n--- {r['uid']} [{r.get('data_source')}] attempt {r.get('accepted_attempt')} "
                      f"gt={r['ground_truth']!r}\n{r['response']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--audit", type=int, default=0, help="print N accepted traces for manual review")
    args = ap.parse_args()

    records = load_last_records(args.traces)
    if not records:
        raise SystemExit("no records found")
    report(records)
    if args.audit:
        audit(records, args.audit)


if __name__ == "__main__":
    main()
