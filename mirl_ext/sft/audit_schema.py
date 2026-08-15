"""Schema + row-anatomy audit of the split parquets (cluster-side, via srun).

Prints per half/family: schema, row count, prompt roles, media-count
histograms, data_source counts, and distinct-label counts -- the facts every
other stage assumes. Read-only.

    srun -p cpu -c 4 --mem=32G <env>/bin/python mirl_ext/sft/audit_schema.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sft_tasks import DATA_ROOT, FAMILIES  # noqa: E402


def audit_file(path: Path) -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    info = {"rows": pf.metadata.num_rows,
            "schema": {f.name: str(f.type)[:80] for f in pf.schema_arrow}}
    batch = next(pf.iter_batches(batch_size=256)).to_pylist()
    r0 = batch[0]
    prompt = r0.get("prompt")
    info["prompt_roles"] = (
        [m.get("role") for m in prompt] if isinstance(prompt, list) else type(prompt).__name__
    )
    for col in ("images", "videos", "signals"):
        lens = [len(r.get(col) or []) for r in batch]
        info[col] = dict(sorted(collections.Counter(lens).items()))
    srcs = collections.Counter(str(r.get("data_source")) for r in batch)
    info["data_sources_first_batch"] = dict(srcs.most_common())
    cols = [c for c in ("data_source", "reward_model") if c in pf.schema_arrow.names]
    full = pq.read_table(path, columns=cols).to_pylist()
    info["data_sources_full"] = dict(collections.Counter(
        str(r.get("data_source")) for r in full).most_common())
    info["distinct_ground_truths"] = len(
        {str((r.get("reward_model") or {}).get("ground_truth")) for r in full}
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--halves", nargs="*", default=["sft", "rl"])
    args = ap.parse_args()

    out = {}
    for half in args.halves:
        for family in args.families:
            path = Path(args.split_root) / half / f"{family}.parquet"
            out[f"{half}/{family}"] = audit_file(path) if path.exists() else "MISSING"
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
