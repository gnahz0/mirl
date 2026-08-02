# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Find label pairs that mean the same thing but are scored as different classes.

Every reward here ends in an exact match after some normalization, so two spellings
of one concept split its examples across two classes and make BOTH unearnable-by-name:
the model cannot know which arbitrary variant a given row uses. That silently caps
macro-F1 and is invisible in a loss curve.

Reported in tiers, loosest last, because the risk of a false positive rises with each:

  1. COLLIDES  -- already identical after that family's own reward normalizer. Not a
                  scoring bug (the reward merges them) but a sign of inconsistent
                  label writing, and it means the raw label space overstates classes.
  2. CASE      -- differ only in capitalization.
  3. SEPARATOR -- differ only in spaces/underscores/hyphens/punctuation.
  4. PLURAL    -- one is the other plus a trailing s/es.
  5. RECIPE    -- same ingredient:percentage mapping, different order/separator
                  (`garlic50_almond50` vs `almond_50_garlic_50`).
  6. NEAR      -- high character similarity, everything else.

CAUTION, learned the hard way: a pair can look identical and be genuinely DISTINCT.
`cloves` (a base food, 6-channel rig) and `clove` (a fragrance extract in the mixture
palette, 4-channel rig) are different substances that share a name -- merging them
would collapse two real classes. So every pair is reported with the ``data_source``\\s
it spans and its row counts, and pairs spanning sources are flagged SEPARATELY as
"check provenance" rather than as defects.

    python scripts/mirl/audit_label_duplicates.py --data-root /work/.../data
"""

from __future__ import annotations

import argparse
import collections
import difflib
import itertools
import re
import sys
from pathlib import Path

FAMILIES = [
    "smellnet_train",
    "ecg_train",
    "haptic_ts_train",
    "climb_train",
    "human_behaviour_train",
    "tactile_train",
]


def norm_case(s: str) -> str:
    return s.strip().lower()


def norm_sep(s: str) -> str:
    return re.sub(r"[\s_\-/,.]+", " ", s.strip().lower()).strip()


def norm_plural(s: str) -> str:
    t = norm_sep(s)
    return re.sub(r"(es|s)$", "", t)


_RECIPE_RE = re.compile(r"([a-z]+)[\s_\-]*(\d+)")


def recipe_key(s: str):
    """Canonical form for '<ingredient><pct>' labels: a sorted (name, pct) mapping.

    A bare token SET is the wrong test for these. It calls
    `apple_10_coriander_90` and `apple_90_coriander_10` identical -- they are
    different recipes -- while missing `almond_50_garlic_50` vs `garlic50_almond50`,
    because splitting on non-alphanumerics turns the latter into {garlic50, almond50}
    rather than {garlic:50, almond:50}. Pair each ingredient WITH its percentage, then
    sort: order- and separator-invariant, but ratio-sensitive.

    Returns None when the label has no percentages, so plain class names fall through
    to the other tiers instead of all collapsing into one bucket.
    """
    # Gate hard on "is this actually a recipe". Without this, free-text labels are
    # parsed as recipes: haptic descriptions saying "around the 2-second mark" yielded
    # 59 bogus groups of unrelated sentences that merely mention similar times.
    if len(s) > 60:
        return None
    pairs = _RECIPE_RE.findall(s.lower())
    if not pairs:
        return None
    total = sum(int(p) for _n, p in pairs)
    if not (95 <= total <= 105):          # recipes are percentages summing to ~100
        return None
    covered = sum(len(n) + len(p) for n, p in pairs)
    if covered < 0.7 * len(re.sub(r"[^a-z0-9]", "", s.lower())):
        return None                        # pairs must account for most of the label
    merged: dict[str, int] = {}
    for name, pct in pairs:
        merged[name] = merged.get(name, 0) + int(pct)
    return tuple(sorted(merged.items()))


def reward_norm(family: str):
    """The family's ACTUAL reward normalizer, so tier 1 reflects real scoring."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        if family.startswith("smellnet"):
            from mirl_ext.rewards.smellnet import _norm_label

            return _norm_label
        if family.startswith("ecg"):
            from mirl_ext.rewards.ecg import _norm

            return _norm
    except Exception:  # noqa: BLE001
        pass
    return norm_case


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", default="/work/mit/ppliang_mit/alecz/data")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--near-threshold", type=float, default=0.92)
    ap.add_argument("--max-labels-for-near", type=int, default=1500,
                    help="skip the O(n^2) NEAR tier above this many labels")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    for family in args.families or FAMILIES:
        src = Path(args.data_root) / f"{family}.parquet"
        if not src.exists():
            print(f"[skip] {src}")
            continue
        rows = pq.read_table(src).to_pylist()
        counts: collections.Counter = collections.Counter()
        sources: dict[str, set] = collections.defaultdict(set)
        for r in rows:
            gt = (r.get("reward_model") or {}).get("ground_truth")
            if gt is None:
                continue
            counts[gt] += 1
            sources[gt].add(r.get("data_source"))
        labels = sorted(counts)

        print("=" * 78)
        print(f"{family}: {len(rows):,} rows, {len(labels):,} distinct labels")

        rnorm = reward_norm(family)
        tiers: dict[str, dict] = {
            "1 COLLIDES (reward already merges)": collections.defaultdict(set),
            "2 CASE only": collections.defaultdict(set),
            "3 SEPARATOR only": collections.defaultdict(set),
            "4 PLURAL only": collections.defaultdict(set),
            "5 SAME RECIPE (order/separator differs, ratios equal)": collections.defaultdict(set),
        }
        for lab in labels:
            tiers["1 COLLIDES (reward already merges)"][rnorm(lab)].add(lab)
            tiers["2 CASE only"][norm_case(lab)].add(lab)
            tiers["3 SEPARATOR only"][norm_sep(lab)].add(lab)
            tiers["4 PLURAL only"][norm_plural(lab)].add(lab)
            rk = recipe_key(lab)
            if rk is not None:
                tiers["5 SAME RECIPE (order/separator differs, ratios equal)"][rk].add(lab)

        seen_pairs: set = set()
        any_found = False
        for tier, groups in tiers.items():
            hits = [sorted(v) for v in groups.values() if len(v) > 1]
            # don't re-report a pair already shown by a stricter tier
            fresh = []
            for grp in hits:
                key = tuple(grp)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                fresh.append(grp)
            if not fresh:
                continue
            any_found = True
            print(f"\n  --- {tier}: {len(fresh)} group(s) ---")
            for grp in sorted(fresh, key=lambda g: -sum(counts[x] for x in g))[:15]:
                srcs = set().union(*(sources[x] for x in grp))
                tag = "  [SPANS SOURCES -> check provenance, may be genuinely distinct]" if len(srcs) > 1 else ""
                detail = ", ".join(f"{x!r}={counts[x]}" for x in grp)
                print(f"    {detail}   sources={sorted(s for s in srcs if s)}{tag}")

        # Block before comparing: an all-pairs scan is O(n^2) and 15k labels would be
        # 100M comparisons, so large families were previously skipped entirely and
        # simply never audited. Two labels this similar must share a rare word, so
        # bucket by rarest token and compare only within buckets.
        df = collections.Counter()
        for lab in labels:
            for w in set(re.split(r"[^a-z0-9]+", lab.lower())):
                if w:
                    df[w] += 1
        buckets: dict[str, list] = collections.defaultdict(list)
        for lab in labels:
            words = [w for w in set(re.split(r"[^a-z0-9]+", lab.lower())) if w]
            if not words:
                continue
            buckets[min(words, key=lambda w: df[w])].append(lab)
        near = []
        checked = 0
        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > 400:
                continue
            for a, b in itertools.combinations(sorted(bucket), 2):
                if tuple(sorted((a, b))) in seen_pairs:
                    continue
                if abs(len(a) - len(b)) > 8:
                    continue
                checked += 1
                r = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
                if r >= args.near_threshold:
                    near.append((r, a, b))
        if True:
            if near:
                any_found = True
                print(f"\n  --- 6 NEAR (ratio >= {args.near_threshold}): {len(near)} pair(s), "
                      f"{checked:,} comparisons after blocking ---")
                for r, a, b in sorted(near, reverse=True)[:15]:
                    srcs = sources[a] | sources[b]
                    tag = "  [SPANS SOURCES]" if len(srcs) > 1 else ""
                    print(f"    {r:.3f}  {a!r}={counts[a]}  vs  {b!r}={counts[b]}{tag}")

        if not any_found:
            print("  no duplicate-looking labels found")


if __name__ == "__main__":
    main()
