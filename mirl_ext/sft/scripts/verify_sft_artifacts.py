"""Verify the exact audited SFT parquet bytes selected for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.sft.artifacts import sha256, verify_audit_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--print-digest",
        action="store_true",
        help="print only the verified audit-manifest SHA-256 (for run identity)",
    )
    args = parser.parse_args()

    verify_audit_manifest(args.parquet_root, args.files)
    digest = sha256(args.parquet_root / "audit_manifest.json")
    if args.print_digest:
        print(digest)
    else:
        print(
            f"verified {len(args.files)} SFT parquet(s) against audit_manifest.json "
            f"({digest[:12]})"
        )


if __name__ == "__main__":
    main()
