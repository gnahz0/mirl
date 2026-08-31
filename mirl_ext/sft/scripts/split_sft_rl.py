"""Split each MIRL family into an SFT part and an RL part (ratio from
config.json ``sft_frac``; currently 50:50, code default 20:80).

GRPO must never be rewarded on something SFT already memorized, and several
families ask many questions about one recording -- so the split unit is a GROUP
(the underlying recording: shared 3DHaptic clip stem for tactile/haptic_ts,
media path otherwise), never a row. Groups are shuffled deterministically and
assigned greedily to whichever side is furthest below its target share,
stratified by (data_source, label). ``sft_frac`` sets the teacher-trace bill:
every SFT row gets a trace (gen_sft_targets.py generates for each row).
Known limitation (recorded in the manifest): ECG has no patient ids, so
patient-level leakage cannot be prevented from these indexes.

    python mirl_ext/sft/scripts/split_sft_rl.py --out-root /work/.../data/split
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import DATA_ROOT, config_path, first_media_path, recording_stem  # noqa: E402

# Target SFT share of rows; lives in config.json ("sft_frac").
SFT_FRAC = float(config_path("sft_frac", "MIRL_SFT_FRAC", "0.2"))

# Group-id mode per family: "path" = media path, "stem" = shared 3DHaptic clip
# stem. Split the data/ veRL indexes (rendered plots) -- NOT trainedve_raw/*,
# which is Stage-1 alignment data that never reaches GRPO.
FAMILIES: dict[str, str] = {
    "smellnet_train": "path",
    "ecg_train": "path",
    "haptic_ts_train": "stem",
    "climb_train": "path",
    "human_behaviour_train": "path",
    "tactile_train": "stem",
}

# Mostly-unique free-text labels: stratify on data_source alone.
LABEL_STRATIFY_EXCLUDE = {"haptic_ts_train", "human_behaviour_train", "tactile_train"}

def _clip_stem(row: dict) -> str:
    """Normalized 3DHaptic recording id, shared between tactile and haptic_ts
    so one physical recording cannot straddle the split."""
    stem = recording_stem(row) or Path(first_media_path(row)).stem
    return f"3dhaptic::{stem}"


def group_id(row: dict, mode: str) -> str:
    if mode == "stem":
        return _clip_stem(row)
    path = first_media_path(row)
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
    sft_frac: float = 0.2,
) -> dict[str, str]:
    """Greedy smallest-half-first per stratum over a seeded shuffle. ``locked``
    (assignments from an earlier family) is authoritative and consumed FIRST --
    without it, 808 shared 3DHaptic clips once straddled the halves."""
    locked = locked or {}
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for gid in groups:
        by_stratum[strata[gid]].append(gid)

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        gids = sorted(by_stratum[stratum])          # sort first: dict order is not a spec
        rng = random.Random(f"{seed}::{stratum}")
        rng.shuffle(gids)
        # Per-stratum tie-break: a fixed one compounds odd-stratum leftovers
        # into a skewed split (measured 54.6/45.4).
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
            # Fill whichever side is furthest below its target share.
            sft_deficit, rl_deficit = n_sft * (1 - sft_frac), n_rl * sft_frac
            to_sft = sft_deficit < rl_deficit or (sft_deficit == rl_deficit and tie_to_sft)
            if to_sft:
                assignment[gid], n_sft = "sft", n_sft + size
            else:
                assignment[gid], n_rl = "rl", n_rl + size
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-root", default=f"{DATA_ROOT}/split")
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

    import pyarrow as pa
    import pyarrow.parquet as pq

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    manifest: dict = {
        "seed": args.seed,
        "sft_frac": SFT_FRAC,
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

    # Pass 1: groups + strata per family, holding only indices.
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
            # Pin multi-stratum groups to the min stratum -- deterministic.
            strata[gid] = min(strata.get(gid, "￿"), stratum_of(row, family))
        plans[family] = (dict(groups), strata)
        del rows

    # Largest family first: it decides shared-group sides, the smaller inherits.
    order = sorted(plans, key=lambda f: -sum(len(v) for v in plans[f][0].values()))
    global_assignment: dict[str, str] = {}

    for family in order:
        mode = families[family]
        groups, strata = plans[family]
        assignment = assign_groups(
            groups, strata, args.seed, locked=global_assignment, sft_frac=SFT_FRAC
        )
        global_assignment.update(assignment)

        idx = {"sft": [], "rl": []}
        for gid, members in groups.items():
            idx[assignment[gid]].extend(members)
        for half in idx:
            idx[half].sort()

        # Straddling is impossible by construction; assert anyway (it broke once).
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
