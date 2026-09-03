"""Shared readers for append-only SFT teacher trace JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def last_records(paths: Iterable[Path]) -> tuple[dict[str, dict], int]:
    """Return the last record for each uid and the number of bad lines.

    Trace generation is append-only, so a later retry supersedes an earlier
    record for the same uid. A killed job can leave a malformed final line;
    malformed and uid-less lines are skipped and counted instead of making an
    otherwise recoverable trace file unreadable.
    """
    records: dict[str, dict] = {}
    skipped = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("uid"), str)
                    or not record["uid"]
                ):
                    skipped += 1
                    continue
                records[record["uid"]] = record
    return records, skipped


def accepted_traces(paths: Iterable[Path]) -> dict[str, dict]:
    """Return last-write-wins records whose final status is accepted.

    Status-less records from legacy and episode-generation traces retain their
    historical meaning of ``accepted``.
    """
    records, skipped = last_records(paths)
    if skipped:
        print(f"[warn] skipped {skipped} unparseable trace line(s)")
    return {
        uid: record
        for uid, record in records.items()
        if record.get("status", "accepted") == "accepted"
    }


def read_status(path: Path) -> dict[str, str]:
    """Return each uid's final status, defaulting legacy records to accepted."""
    if not path.exists():
        return {}
    records, _ = last_records([path])
    return {uid: record.get("status", "accepted") for uid, record in records.items()}
