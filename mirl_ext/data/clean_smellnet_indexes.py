#!/usr/bin/env python3
"""Build leakage-checked, content-deduplicated SmellNet alignment indexes."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.data.schema import DATA_ROOT  # noqa: E402

import pyarrow as pa
import pyarrow.parquet as pq

from mirl_ext.data.signals import load_signal_csv


def canonical_label(value: object) -> str:
    """Turn filename-style labels into stable SigLIP-friendly text."""
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    return " ".join(text.split())


def numerical_fingerprint(path: Path) -> str:
    """Hash float32 sensor values while deliberately excluding timestamp columns."""
    tensor = load_signal_csv(str(path))
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clean_rows(table: pa.Table) -> tuple[list[dict], Counter[str], int]:
    """Canonicalize labels and retain the first row for each numerical recording."""
    cleaned: list[dict] = []
    seen: set[str] = set()
    labels: Counter[str] = Counter()
    duplicate_count = 0
    for row in table.to_pylist():
        signals = row.get("signals") or []
        if len(signals) != 1 or not signals[0].get("signal"):
            raise ValueError("expected exactly one SmellNet signal path per row")
        fingerprint = numerical_fingerprint(Path(signals[0]["signal"]))
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        reward = dict(row.get("reward_model") or {})
        reward["ground_truth"] = canonical_label(reward.get("ground_truth"))
        row["reward_model"] = reward
        labels[reward["ground_truth"]] += 1
        # Keep this transient audit value out of the persisted dataset schema.
        row["__numerical_fingerprint"] = fingerprint
        cleaned.append(row)
    return cleaned, labels, duplicate_count


def write_clean(data_root: Path) -> None:
    outputs: dict[str, list[dict]] = {}
    schemas: dict[str, pa.Schema] = {}
    for split in ("train", "valid"):
        source = data_root / f"smellnet_{split}.parquet"
        table = pq.read_table(source)
        rows, labels, duplicate_count = clean_rows(table)
        outputs[split] = rows
        schemas[split] = table.schema
        print(
            f"{split}: {table.num_rows} -> {len(rows)} rows; "
            f"removed_duplicates={duplicate_count}; labels={len(labels)}"
        )

    train_hashes = {row["__numerical_fingerprint"] for row in outputs["train"]}
    valid_hashes = {row["__numerical_fingerprint"] for row in outputs["valid"]}
    overlap = train_hashes & valid_hashes
    if overlap:
        raise RuntimeError(f"SmellNet train/valid numerical leakage: {len(overlap)} hashes")

    for split in ("train", "valid"):
        destination = data_root / f"smellnet_{split}_clean.parquet"
        temporary = data_root / f".{destination.name}.tmp"
        persisted = []
        for row in outputs[split]:
            row = dict(row)
            row.pop("__numerical_fingerprint")
            persisted.append(row)
        clean_table = pa.Table.from_pylist(persisted, schema=schemas[split])
        pq.write_table(clean_table, temporary, compression="snappy")
        temporary.replace(destination)
        print(f"wrote {clean_table.num_rows} rows: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(f"{DATA_ROOT}/trainedve_raw"),
    )
    args = parser.parse_args()
    write_clean(args.data_root)


if __name__ == "__main__":
    main()
