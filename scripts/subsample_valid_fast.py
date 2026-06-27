#!/usr/bin/env python3
"""Build *_valid_fast.parquet: stratified per-data_source subsamples of the slow valid
splits so a full validation pass fits in ~1h.

Validation time is dominated by VIDEO decode (tactile 9407 + human_behaviour 2000 videos);
image splits (ecg, climb, smellnet, haptic) decode cheaply and are kept full. We cap rows
per data_source so every task's metric still has a sample, then write new parquet files
(schema/media refs preserved via take()). Run from the repo so paths resolve; output -> data/.
"""
import os
import random
from collections import defaultdict

import pyarrow.parquet as pq

random.seed(0)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DATA = os.path.abspath(DATA)

# (input split, output split, per-data_source cap). Only video-heavy splits are capped.
JOBS = [
    ("tactile_valid", "tactile_valid_fast", 40),
    ("human_behaviour_valid_small", "human_behaviour_valid_fast", 40),
]


def subsample(infile: str, outfile: str, cap: int) -> None:
    t = pq.read_table(infile)
    ds = t.column("data_source").to_pylist()
    buckets = defaultdict(list)
    for i, s in enumerate(ds):
        buckets[s].append(i)
    keep = []
    for s, idxs in buckets.items():
        keep.extend(idxs if len(idxs) <= cap else random.sample(idxs, cap))
    keep.sort()
    out = t.take(keep)
    pq.write_table(out, outfile)
    print(f"{os.path.basename(outfile):34} {t.num_rows:6d} -> {out.num_rows:5d} "
          f"({len(buckets)} data_sources, cap={cap})")


def main() -> None:
    for src, dst, cap in JOBS:
        subsample(os.path.join(DATA, f"{src}.parquet"),
                  os.path.join(DATA, f"{dst}.parquet"), cap)


if __name__ == "__main__":
    main()
