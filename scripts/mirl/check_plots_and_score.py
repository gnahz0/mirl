# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Verify the plot PNGs are real images, and score blind predictions per class.

Two questions the trace files cannot answer on their own:

1. **Are the plots actually images?** A file can exist, be a valid PNG, and still be
   blank or single-colour -- in which case "grounded" generation is describing
   nothing. Checks decodability, size, and pixel variance, and reports the darkest /
   least-varied outliers rather than just a mean.

2. **How often is the answer right when it is NOT supplied?** The SFT traces are
   answer-conditioned, so their boxed answer matches by construction and accuracy is
   trivially 100%. Blind predictions (``gen_sft_targets.py --blind``) are the honest
   signal, and per-class F1 shows whether a family is genuinely readable or whether
   one majority class is carrying the average.

    python scripts/mirl/check_plots_and_score.py --tasks ts_sft_tasks.jsonl \\
        --image-root data/sft/ts_images --preds blind_preds.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def resolve(image_path: str, image_root: Path | None) -> Path:
    p = Path(image_path)
    if image_root is not None:
        cand = image_root / p.parent.name / p.name
        if cand.is_file():
            return cand
    return p


def check_images(tasks: list[dict], image_root: Path | None, sample: int) -> None:
    from PIL import Image
    from PIL import ImageStat

    print("=" * 78)
    print("PLOT IMAGE VERIFICATION")
    print("=" * 78)
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for t in tasks:
        by_family[t["family"]].append(t)

    for family, items in sorted(by_family.items()):
        subset = items[:sample] if sample else items
        missing = decoded = blank = 0
        sizes, kb, stds, modes = [], [], [], collections.Counter()
        worst = []
        for t in subset:
            path = resolve(t.get("image_path", ""), image_root)
            if not path.is_file():
                missing += 1
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    sizes.append(im.size)
                    modes[im.mode] += 1
                    g = im.convert("L")
                    st = ImageStat.Stat(g)
                    sd = st.stddev[0]
                decoded += 1
                kb.append(path.stat().st_size / 1024)
                stds.append(sd)
                worst.append((sd, path.name))
                # A real plot has ink on background; near-zero variance means a blank
                # canvas, which would make "grounded" reasoning pure invention.
                if sd < 3.0:
                    blank += 1
            except Exception as exc:  # noqa: BLE001
                print(f"    UNDECODABLE {path.name}: {type(exc).__name__}")

        def med(x):
            return sorted(x)[len(x) // 2] if x else float("nan")

        print(f"\n{family}  (checked {len(subset)} of {len(items)})")
        print(f"  decodable      : {decoded}/{len(subset)}   missing: {missing}")
        print(f"  modes          : {dict(modes)}")
        print(f"  distinct sizes : {len(set(sizes))}  e.g. {sorted(set(sizes))[:3]}")
        if not kb:
            # All missing: almost always a wrong/absent --image-root rather than
            # corrupt data. Say so instead of dying on min() of an empty list.
            print("  NO IMAGES RESOLVED -- check --image-root (paths are cluster-absolute)")
            continue
        print(f"  file KB        : median {med(kb):.0f}  min {min(kb):.0f}  max {max(kb):.0f}")
        print(f"  pixel stddev   : median {med(stds):.1f}  min {min(stds):.1f}")
        print(f"  blank/near-flat (stddev<3): {blank}")
        worst.sort()
        print(f"  least-varied   : {[(round(s,1), n) for s, n in worst[:3]]}")


def prf1(pred: list[str], gold: list[str]) -> dict:
    """Per-class precision/recall/F1 plus macro/micro, on exact normalized match."""
    classes = sorted(set(gold))
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    for p, g in zip(pred, gold, strict=True):
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1          # counted even if p is not a valid class
    rows = []
    for c in classes:
        p_den = tp[c] + fp[c]
        r_den = tp[c] + fn[c]
        prec = tp[c] / p_den if p_den else 0.0
        rec = tp[c] / r_den if r_den else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((c, tp[c], r_den, prec, rec, f1))
    macro_f1 = sum(r[5] for r in rows) / len(rows) if rows else 0.0
    acc = sum(tp.values()) / len(gold) if gold else 0.0
    return {"rows": rows, "macro_f1": macro_f1, "accuracy": acc, "n": len(gold)}


def family_modes(tasks: list[dict]) -> dict:
    """Decide class-vs-free-text per family from the FULL label space, once.

    Deciding it from whatever sample happens to be scored is a bug: smellnet has 60
    distinct labels among 90 rows (-> "free text") but 94 among 242 (-> "classes"),
    so the same family scored two different ways and the numbers were not comparable
    across runs. The label space is a property of the family, not of the sample.
    """
    by_family: dict[str, set] = collections.defaultdict(set)
    counts: dict[str, int] = collections.Counter()
    for t in tasks:
        by_family[t["family"]].add(norm(t["ground_truth"]))
        counts[t["family"]] += 1
    modes = {}
    for fam, labels in by_family.items():
        modes[fam] = "free_text" if len(labels) > 0.6 * counts[fam] else "classes"
    return modes


def score(preds: list[dict], top: int, modes: dict | None = None) -> None:
    print("\n" + "=" * 78)
    print("BLIND PREDICTION ACCURACY (answer withheld -- the honest test)")
    print("=" * 78)
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for p in preds:
        by_family[p["family"]].append(p)

    for family, items in sorted(by_family.items()):
        gold = [norm(x["ground_truth"]) for x in items]
        pred = [norm(x["predicted"]) for x in items]
        n_classes = len(set(gold))

        # Free-text families (one unique label per row) cannot be scored as classes;
        # exact match would be ~0 and per-class F1 meaningless. Use token overlap,
        # which is what their reward module actually uses. Mode comes from the family's
        # FULL label space when available, never from this sample.
        mode = (modes or {}).get(family)
        if mode is None:
            mode = "free_text" if n_classes > 0.6 * len(items) else "classes"
        if mode == "free_text":
            def toks(s):
                return {w for w in re.split(r"[^a-z0-9]+", s) if w}

            f1s = []
            for p, g in zip(pred, gold, strict=True):
                tp_, gp = toks(p), toks(g)
                inter = len(tp_ & gp)
                prec = inter / len(tp_) if tp_ else 0.0
                rec = inter / len(gp) if gp else 0.0
                f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            print(f"\n{family}  n={len(items)}  ({n_classes} distinct labels -> free text)")
            print(f"  exact match      : {sum(1 for p, g in zip(pred, gold) if p == g)}/{len(items)}")
            print(f"  mean token-F1    : {sum(f1s) / len(f1s):.3f}")
            continue

        res = prf1(pred, gold)
        maj = collections.Counter(gold).most_common(1)[0]
        print(f"\n{family}  n={res['n']}  classes={n_classes}")
        print(f"  accuracy   : {res['accuracy']:.3f}")
        print(f"  macro F1   : {res['macro_f1']:.3f}")
        print(f"  baselines  : majority={maj[1] / res['n']:.3f} ('{maj[0][:30]}')  uniform={1 / n_classes:.3f}")
        print(f"  margin over majority : {res['accuracy'] - maj[1] / res['n']:+.3f}")
        rows = sorted(res["rows"], key=lambda r: -r[2])[:top]
        print(f"\n  {'class':38s} {'n':>5s} {'TP':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
        for c, tp_, support, prec, rec, f1 in rows:
            print(f"  {c[:36]:38s} {support:5d} {tp_:4d} {prec:6.3f} {rec:6.3f} {f1:6.3f}")
        off = collections.Counter(
            p for p, g in zip(pred, gold, strict=True) if p != g and p not in set(gold)
        )
        if off:
            print(f"  predictions outside the label set: {sum(off.values())} e.g. {off.most_common(3)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", help="task JSONL (for image verification)")
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=300, help="images to check per family (0=all)")
    ap.add_argument("--check-images", action="store_true", help="verify plots even without --image-root")
    ap.add_argument("--preds", help="blind prediction JSONL from --blind")
    ap.add_argument("--compare", help="second prediction JSONL to score alongside --preds")
    ap.add_argument("--top", type=int, default=12, help="classes to list per family")
    args = ap.parse_args()

    tasks = []
    if args.tasks:
        tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
        if args.image_root or args.check_images:
            check_images(tasks, args.image_root, args.sample)
    modes = family_modes(tasks) if args.tasks else None
    if modes:
        print(f"\nscoring mode per family (from full label space): {modes}")
    if args.preds:
        preds = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]
        print(f"\n######## {Path(args.preds).name} ########")
        score(preds, args.top, modes)
    if args.compare:
        other = [json.loads(l) for l in Path(args.compare).read_text().splitlines() if l.strip()]
        print(f"\n######## {Path(args.compare).name} ########")
        score(other, args.top, modes)


if __name__ == "__main__":
    main()
