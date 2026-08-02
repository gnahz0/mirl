# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Split each MIRL family 50:50 into an SFT half and an RL half.

The point of the split is that GRPO must never be rewarded on something the SFT
cold-start already memorized. A plain row-level random split does not achieve that
here, because several families ask MANY questions about ONE underlying recording:

    tactile           28,164 rows over  1,983 videos   (~14 questions per video)
    human_behaviour   65,605 rows over 35,426 clips    (~1.9 questions per clip)
    haptic_ts          1,575 rows over  1,575 clips    (1:1, but SHARES the
                                                        underlying clips with tactile)

Splitting those by row puts two questions about the same video on opposite sides.
The model then meets that video in SFT and is scored on it again in RL, which is
exactly the leakage this split exists to prevent — and it is silent, because both
rows look like distinct examples.

So the split unit is a GROUP (one underlying recording), never a row:

  * ``tactile``/``haptic_ts`` -> the shared clip stem, so a clip that appears in
    both families lands on the SAME side in both. Cross-family leakage is real
    here: `tactile.extra_info.video_path` and `haptic_ts.extra_info.stem` name the
    same 3DHaptic recordings.
  * everything else -> the media path (signal/image/video).

Within a stratum, groups are shuffled deterministically (seed) and assigned
greedily to whichever half currently holds fewer ROWS, so unequal group sizes
still yield a near-exact 50/50 row split.

KNOWN LIMITATION, recorded in the manifest: ECG rows carry only a record id
(`HR00521.pt`), not a patient id, and PTB-XL contains multiple ECGs per patient
(~21.8k records from ~18.9k patients). Patient-level leakage therefore CANNOT be
prevented from these indexes. It is bounded (~13% of PTB-XL records share a
patient with another record) but it is not zero, and it would need the original
PTB-XL metadata to fix.

    python scripts/mirl/split_sft_rl.py --data-root /work/.../trainedve_raw \\
        --out-root /work/.../data/split
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

# Families to split, and how to derive a row's group id.
#   "path"  -> the media path (one recording per row)
#   "stem"  -> the shared 3DHaptic clip stem (tactile + haptic_ts must agree)
# Default = the ORIGINAL veRL indexes under data/, which is what GRPO actually
# consumes (see run_qwen35_grpo.sh: train_files reads data/<family>_train.parquet).
# Those rows carry the rendered plot PNG in `images`; the trainedve_raw/* indexes are
# the Stage-1 ALIGNMENT data (raw signal paths, no images) and are NOT what RL trains
# on -- splitting those would produce halves that never reach GRPO.
FAMILIES: dict[str, str] = {
    "smellnet_train": "path",
    "ecg_train": "path",
    "haptic_ts_train": "stem",
    "climb_train": "path",
    "human_behaviour_train": "path",
    "tactile_train": "stem",
}

# Families whose labels are mostly unique free text (one row per label). Stratifying
# on such a label is meaningless -- every stratum would have one member and the
# "balance" would be an illusion -- so stratify on data_source alone.
LABEL_STRATIFY_EXCLUDE = {"haptic_ts_train", "human_behaviour_train", "tactile_train"}

_IDX_SUFFIX = re.compile(r"_idx\d+$")


def _media_path(row: dict) -> str:
    for key in ("signals", "images", "videos"):
        entries = row.get(key)
        if entries:
            entry = entries[0]
            if isinstance(entry, dict):
                for field in ("signal", "image", "video", "path"):
                    if entry.get(field):
                        return str(entry[field])
                return json.dumps(entry, sort_keys=True)
            return str(entry)
    return ""


def _extra(row: dict) -> dict:
    ei = row.get("extra_info")
    if isinstance(ei, str):
        try:
            ei = json.loads(ei)
        except json.JSONDecodeError:
            return {}
    return ei if isinstance(ei, dict) else {}


def _clip_stem(row: dict) -> str:
    """Normalized 3DHaptic recording id, shared between tactile and haptic_ts.

    tactile carries `video_path` like
      reencoded/visual-tactile/2025-10-10_arnie_recognition_grasp_boardA_idx0.mp4
    haptic_ts carries `stem` like
      2025-10-10_arnie_recognition_grasp_boardA_idx0
    Both reduce to the same key, so one physical recording cannot straddle the split.
    The trailing `_idx<N>` is kept: it distinguishes separate takes, not questions.
    """
    ei = _extra(row)
    stem = ei.get("stem")
    if not stem:
        vp = ei.get("video_path") or _media_path(row)
        stem = Path(str(vp)).stem
    return f"3dhaptic::{stem}"


def group_id(row: dict, mode: str) -> str:
    if mode == "stem":
        return _clip_stem(row)
    path = _media_path(row)
    return f"path::{path}" if path else f"row::{id(row)}"


def stratum_of(row: dict, family: str) -> str:
    ds = str(row.get("data_source", ""))
    if family in LABEL_STRATIFY_EXCLUDE:
        return ds
    label = (row.get("reward_model") or {}).get("ground_truth")
    return f"{ds}||{label}"


def assign_groups(
    groups: dict[str, list[int]],
    strata: dict[str, str],
    seed: int,
    locked: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign each group to 'sft' or 'rl', balancing ROW counts inside each stratum.

    Greedy smallest-half-first over a deterministically shuffled group order. Groups
    stay intact; only their placement varies, so row balance is approximate but tight
    (exact when every group is one row, which is the common case).

    ``locked`` carries assignments already made by an EARLIER family and is
    authoritative: a 3DHaptic clip that tactile placed in the SFT half must go to the
    SFT half in haptic_ts too. Without this the shared clip stem is cosmetic --
    assigning each family independently let 808 clips straddle the halves, which the
    per-family disjointness check cannot see because it never compares families.
    Locked groups are consumed FIRST so their rows are counted before any free choice
    is made, otherwise the greedy balance would be computed against a stale total.
    """
    locked = locked or {}
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for gid in groups:
        by_stratum[strata[gid]].append(gid)

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        gids = sorted(by_stratum[stratum])          # sort first: dict order is not a spec
        rng = random.Random(f"{seed}::{stratum}")
        rng.shuffle(gids)
        # Alternate which side wins ties, per stratum. A fixed tie-break looks
        # harmless but is systematic: every odd-sized stratum hands its extra row to
        # the same half, and with 132 smellnet strata that compounded into a 54.6/45.4
        # split. Deciding per stratum makes the leftovers cancel instead of accumulate.
        tie_to_sft = rng.random() < 0.5
        n_sft = n_rl = 0
        free = []
        for gid in gids:
            side = locked.get(gid)
            if side is None:
                free.append(gid)
                continue
            assignment[gid] = side
            if side == "sft":
                n_sft += len(groups[gid])
            else:
                n_rl += len(groups[gid])
        for gid in free:
            size = len(groups[gid])
            to_sft = n_sft < n_rl or (n_sft == n_rl and tie_to_sft)
            if to_sft:
                assignment[gid], n_sft = "sft", n_sft + size
            else:
                assignment[gid], n_rl = "rl", n_rl + size
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", default="/work/mit/ppliang_mit/alecz/data")
    ap.add_argument("--out-root", default="/work/mit/ppliang_mit/alecz/data/split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="subset of FAMILIES to split (default: all six). Splitting a SUBSET is "
        "safe only if it includes every family sharing a group key -- tactile and "
        "haptic_ts share 3DHaptic clips, so split them together or not at all.",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    families = FAMILIES if not args.families else {f: FAMILIES[f] for f in args.families}
    stem_fams = {f for f, m in FAMILIES.items() if m == "stem"}
    if args.families and stem_fams - set(args.families):
        missing = sorted(stem_fams - set(args.families))
        print(
            f"WARNING: splitting a subset that omits {missing}, which share 3DHaptic "
            "clip ids with the selected families. Those clips can then land on "
            "opposite sides across the two runs. Re-split them together to be safe."
        )

    import pyarrow.parquet as pq

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    manifest: dict = {
        "seed": args.seed,
        "split_unit": "group (underlying recording), not row",
        "families": {},
        "known_limitations": [
            "ECG rows carry a record id, not a patient id; PTB-XL has multiple ECGs "
            "per patient (~21.8k records / ~18.9k patients), so patient-level leakage "
            "between the SFT and RL halves cannot be prevented from these indexes.",
            "SmellNet records the same substance across several sessions/days. Those "
            "are separate recordings and may land on opposite sides; that is intended "
            "(labels repeat by design) but it is NOT full independence.",
        ],
    }

    # Pass 1: derive groups + strata per family, holding only indices (not rows), so
    # all six families fit in memory at once and the assignment can be made globally.
    plans: dict[str, tuple[dict[str, list[int]], dict[str, str]]] = {}
    for family, mode in families.items():
        src = data_root / f"{family}.parquet"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        rows = pq.read_table(src).to_pylist()
        groups: dict[str, list[int]] = collections.defaultdict(list)
        strata: dict[str, str] = {}
        for i, row in enumerate(rows):
            gid = group_id(row, mode)
            groups[gid].append(i)
            # A group spanning several strata (e.g. one video with questions of
            # different data_source) is pinned to the FIRST stratum seen, sorted, so
            # the choice is deterministic rather than dependent on row order.
            strata[gid] = min(strata.get(gid, "￿"), stratum_of(row, family))
        plans[family] = (dict(groups), strata)
        del rows

    # Assign in descending row count so the family with the most rows riding on a
    # shared group decides its side; the smaller one inherits. (tactile has 28k rows
    # over the same 3DHaptic clips that haptic_ts covers with 1.5k, so letting
    # haptic_ts win those coin flips would distort the larger family's balance.)
    order = sorted(plans, key=lambda f: -sum(len(v) for v in plans[f][0].values()))
    global_assignment: dict[str, str] = {}

    for family in order:
        mode = families[family]
        groups, strata = plans[family]
        assignment = assign_groups(groups, strata, args.seed, locked=global_assignment)
        # Only stem-keyed groups are shared across families; path-keyed ids are unique
        # per family anyway, but recording them costs nothing and keeps the rule simple.
        global_assignment.update(assignment)

        idx = {"sft": [], "rl": []}
        for gid, members in groups.items():
            idx[assignment[gid]].extend(members)
        for half in idx:
            idx[half].sort()

        # Each group carries exactly one side, so straddling is impossible by
        # construction here; assert it anyway because the cross-FAMILY version of this
        # invariant did silently break once (808 3DHaptic clips), and the per-family
        # check is what failed to notice.
        sides = collections.Counter(assignment[gid] for gid in groups)
        assert sides["sft"] + sides["rl"] == len(groups), f"{family}: unassigned groups"

        n_rows = sum(len(v) for v in groups.values())
        info = {
            "rows": n_rows,
            "groups": len(groups),
            "rows_per_group": round(n_rows / max(1, len(groups)), 2),
            "group_key": mode,
            "strata": len(set(strata.values())),
            "sft_rows": len(idx["sft"]),
            "rl_rows": len(idx["rl"]),
            "sft_frac": round(len(idx["sft"]) / max(1, n_rows), 4),
        }
        manifest["families"][family] = info
        print(
            f"{family:26s} rows={info['rows']:7d} groups={info['groups']:7d} "
            f"({info['rows_per_group']:5.2f} rows/group, key={mode:4s}) "
            f"-> sft {info['sft_rows']:7d} / rl {info['rl_rows']:7d} "
            f"({info['sft_frac']:.1%} sft)"
        )

        if args.dry_run:
            continue
        import pyarrow as pa

        table = pq.read_table(data_root / f"{family}.parquet")
        for half in ("sft", "rl"):
            dest = out_root / half
            dest.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.take(pa.array(idx[half])), dest / f"{family}.parquet")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nmanifest -> {out_root / 'split_manifest.json'}")
    total_sft = sum(f["sft_rows"] for f in manifest["families"].values())
    total_rl = sum(f["rl_rows"] for f in manifest["families"].values())
    print(f"TOTAL sft={total_sft} rl={total_rl}")


if __name__ == "__main__":
    main()
