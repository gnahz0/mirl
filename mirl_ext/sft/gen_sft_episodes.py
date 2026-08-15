"""Few-shot prediction episodes: SmellNet SFT traces the teacher must EARN.

The teacher is never given the answer: each episode shows N candidate classes
with 2-3 labelled support plots each plus one unlabelled query, and the trace is
kept only if the prediction matches ground truth and is self-contained (the
student sees only the original 50-way prompt + query plot, so the episode
scaffolding must never be cited). Planning is deterministic from --seed.

    # laptop (endpoint access + staged plots); resume-by-uid, re-runnable
    python mirl_ext/sft/gen_sft_episodes.py --out data/sft/ts_traces_episodes.jsonl [--dry-run|--limit 10]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.rewards._common import extract_boxed_answer, format_reward  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_sft_targets import (  # noqa: E402
    DEFAULT_MODEL,
    _LEAK_RE,
    _img,
    _norm,
    _resolve_image,
    backoff,
    make_client,
    read_done_uids,
)

DATA_SOURCE = "smellnet_base"

# Upstream's canonical taxonomy (MIT-MI/SmellNet models/utils.py, 10/category),
# used for hard negatives until the trained encoder provides similarity.
CATEGORY = {
    # Nuts
    "almond": "nuts", "brazil_nut": "nuts", "cashew": "nuts", "chestnuts": "nuts",
    "hazelnut": "nuts", "peanuts": "nuts", "pecans": "nuts", "pili_nut": "nuts",
    "pistachios": "nuts", "walnuts": "nuts",
    # Spices
    "allspice": "spices", "chervil": "spices", "cinnamon": "spices", "cloves": "spices",
    "cumin": "spices", "ginger": "spices", "mustard": "spices", "nutmeg": "spices",
    "saffron": "spices", "star_anise": "spices",
    # Herbs
    "angelica": "herbs", "chamomile": "herbs", "chives": "herbs", "coriander": "herbs",
    "dill": "herbs", "garlic": "herbs", "mint": "herbs", "mugwort": "herbs",
    "oregano": "herbs", "turnip": "herbs",
    # Fruits
    "apple": "fruits", "banana": "fruits", "kiwi": "fruits", "lemon": "fruits",
    "mandarin_orange": "fruits", "mango": "fruits", "peach": "fruits", "pear": "fruits",
    "pineapple": "fruits", "strawberry": "fruits",
    # Vegetables
    "asparagus": "vegetables", "avocado": "vegetables", "broccoli": "vegetables",
    "brussel_sprouts": "vegetables", "cabbage": "vegetables", "cauliflower": "vegetables",
    "potato": "vegetables", "radish": "vegetables", "sweet_potato": "vegetables",
    "tomato": "vegetables",
}
ALL_CLASSES = sorted(CATEGORY)
assert len(ALL_CLASSES) == 50, f"CATEGORY has {len(ALL_CLASSES)} classes, expected 50"

# Self-containment filter on top of _LEAK_RE: the student never sees the episode
# scaffolding, so citing it is a leak. Bare "support(s)"/"example" deliberately
# spared ("this supports garlic", "for example"). Subsumes gen_sft_targets'
# _GALLERY_LEAK_RE word list -- keep in sync.
_EPISODE_LEAK_RE = re.compile(
    r"\b(support(s|ing)?[ -](set|plot|image|example|recording)s?|"
    r"demonstrations?|galler(y|ies)|references?|examples|candidates?|"
    r"shortlist|labell?ed (plot|image|recording)s?|"
    r"(class|option|substance) [A-E]\b|"
    r"(shown|provided|given|presented|attached) (above|earlier|previously)|"
    r"(first|second|third|fourth|fifth|last) (plot|image|panel)|"
    r"quer(y|ies))\b",
    re.IGNORECASE,
)

# Opening sentence and think/boxed instruction are VERBATIM from the student's
# smellnet system prompt, so the teacher writes in the student's framing.
SYSTEM_PROMPT = (
    "You are an expert in analyzing sensor data from electronic noses (e-nose). "
    "The provided images show time-series readings from multiple gas sensors "
    "(each subplot is a different sensor channel: NO2, C2H5OH, VOC, CO, "
    "Alcohol, LPG).\n\n"
    "You will be shown, for several candidate substances, a short substance "
    "profile and a few LABELLED recordings each, followed by one UNLABELLED "
    "recording. Decide which candidate substance the unlabelled recording "
    "shows. Treat the profiles as your own background knowledge -- never cite "
    "them as provided text. What distinguishes "
    "substances is the JOINT pattern across channels -- relative rise rates, "
    "plateau levels, which channels respond and in what order, response shape "
    "-- not absolute values, since baselines drift between sessions.\n\n"
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed "
    "within <think> </think> tags. The final answer MUST BE wrapped in "
    "\\boxed{}. Output EXACTLY this shape and nothing else:\n"
    "   <think> reasoning </think> \\boxed{substance}\n\n"
    "Strict rules for the reasoning (a violation voids the output):\n"
    "1. The model trained on your reasoning will see the unlabelled plot ALONE "
    "-- no labelled recordings, no candidate list. Write the reasoning as if "
    "you identified the substance from that one plot using general knowledge "
    "of how substances present to a gas sensor.\n"
    "2. NEVER mention the setup. Do not write: 'examples', 'references', "
    "'support set', 'demonstration', 'gallery', 'candidates', 'query', 'shown "
    "above', 'the labelled recordings', 'Class A', or 'the first/second "
    "image'. Call the unlabelled recording 'this recording' or 'the plot'.\n"
    "3. Express comparisons as class knowledge, e.g. 'garlic typically drives "
    "a sharp early rise in VOC and Alcohol with a high sustained plateau' -- "
    "never 'this matches the garlic recordings shown'.\n"
    "4. Structure: concrete observations of THIS plot (which channels move, "
    "when, how much) -> the typical signatures of the plausible substances, "
    "explicitly ruling out the closest confuser -> conclusion. 3-6 sentences. "
    "Concise beats florid.\n"
    "5. \\boxed{} must contain exactly one substance name, verbatim as given.\n"
)

QUERY_INSTRUCTION = (
    "Which substance is this recording? "
    "Follow the output shape and reasoning rules exactly."
)

_print_lock = threading.Lock()


def load_tasks(path: Path, image_root: Path | None) -> list[dict]:
    """Parse, filter to smellnet_base, resolve every plot path up front (fail loudly)."""
    tasks, missing = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("data_source") != DATA_SOURCE:
            continue
        # Current exporter writes image_paths (a list); older files image_path.
        rec["image_path"] = rec.get("image_path") or (rec.get("image_paths") or [""])[0]
        resolved = _resolve_image(rec["image_path"], image_root)
        if resolved is None:
            missing.append(rec["image_path"])
        else:
            rec["resolved_image"] = str(resolved)
            tasks.append(rec)
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} plots not found (first 5: {missing[:5]}). "
            "Stage them or pass --image-root."
        )
    return tasks


def _hard_pool(y):
    """Same-category peers -- the definition of a hard negative."""
    return [c for c in ALL_CLASSES if CATEGORY[c] == CATEGORY[y] and c != y]


def _rand_pool(y, hard):
    return [c for c in ALL_CLASSES if c != y and c not in hard]


def sample_candidates(y, counter, rng, n_hard, n_rand):
    """Least-used eligible class per negative slot (rng tie-break) so all 50
    classes accumulate negative coverage. Single-threaded; counter unlocked."""

    def least_used(pool, k):
        pool = list(pool)
        rng.shuffle(pool)                       # rng tie-break under the sort
        pool.sort(key=lambda c: counter[c])     # stable: keeps the shuffle order
        return pool[:k]

    hard = least_used(_hard_pool(y), n_hard)
    rand = least_used(_rand_pool(y, hard), n_rand + max(0, n_hard - len(hard)))
    for c in hard + rand:
        counter[c] += 1
    return hard, rand


def sample_supports(cls, pool_by_class, shots, rng):
    pool = pool_by_class[cls]
    return rng.sample(pool, min(shots, len(pool)))


def make_episode(task, negatives_hard, negatives_rand, pool_by_class, shots, rng):
    y = task["ground_truth"]
    order = [y] + negatives_hard + negatives_rand
    rng.shuffle(order)                          # y must not be positionally biased
    return {
        "task": task,
        "y": y,
        "order": order,
        "hard": negatives_hard,
        "supports": {c: sample_supports(c, pool_by_class, shots, rng) for c in order},
    }


def plan_episodes(queries, args) -> list[dict]:
    """Deterministic from --seed; all sampling happens here, before the pool."""
    counter = collections.Counter({c: 0 for c in ALL_CLASSES})
    episodes = []
    for task in sorted(queries, key=lambda t: t["uid"]):
        rng = random.Random(f"{args.seed}::{task['uid']}")
        hard, rand = sample_candidates(
            task["ground_truth"], counter, rng, args.hard_negatives,
            args.n_way - 1 - args.hard_negatives,
        )
        episodes.append(make_episode(task, hard, rand, args.pool_by_class, args.shots, rng))
    uses = [counter[c] for c in ALL_CLASSES]
    print(
        f"planned {len(episodes)} episodes | negative uses/class "
        f"min={min(uses)} max={max(uses)} | classes never negative: "
        f"{sum(1 for u in uses if u == 0)}"
    )
    return episodes


def resample_episode(ep, args, attempt) -> dict:
    """Fresh negatives/supports/order for late retries; y kept; local rng."""
    task = ep["task"]
    rng = random.Random(f"{args.seed}::{task['uid']}::retry{attempt}")
    y = task["ground_truth"]
    hard_pool = _hard_pool(y)
    hard = rng.sample(hard_pool, min(args.hard_negatives, len(hard_pool)))
    rand = rng.sample(_rand_pool(y, hard), args.n_way - 1 - len(hard))
    return make_episode(task, hard, rand, args.pool_by_class, args.shots, rng)


def build_episode_content(ep, descriptions=None) -> list[dict]:
    """One interleaved user turn: per-class blocks (profile + labelled plots),
    then the query."""
    parts = [{
        "type": "text",
        "text": (
            f"Labelled e-nose recordings for {len(ep['order'])} substances, "
            "followed by one unlabelled recording."
        ),
    }]
    for cls in ep["order"]:
        rows = ep["supports"][cls]
        header = f"=== Substance: {cls} — {len(rows)} labelled recordings ==="
        if descriptions and cls in descriptions:
            header += f"\nProfile: {descriptions[cls]}"
        parts.append({"type": "text", "text": header})
        parts.extend(_img(r["resolved_image"]) for r in rows)
    parts.append({"type": "text", "text": "=== UNLABELLED RECORDING ==="})
    parts.append(_img(ep["task"]["resolved_image"]))
    parts.append({"type": "text", "text": QUERY_INSTRUCTION})
    return parts


def validate_episode(text: str, ground_truth: str) -> tuple[bool, str, str | None]:
    """(accepted, reason, predicted) -- GRPO-reward gates + self-containment."""
    if not text:
        return False, "empty", None
    if format_reward(text) != 1.0:
        return False, "format", None
    boxed = extract_boxed_answer(text)
    if boxed is None:
        return False, "no_boxed", None
    if _norm(boxed) != _norm(ground_truth):
        return False, "wrong", _norm(boxed)
    if _LEAK_RE.search(text):
        return False, "oracle_leak", None
    if _EPISODE_LEAK_RE.search(text):
        return False, "episode_leak", None
    return True, "ok", _norm(boxed)


def generate_one(client, ep0: dict, args, stats: dict) -> dict | None:
    """One episode -> one accepted trace, or None. Attempts 0-1 reuse the episode;
    2+ resample negatives (a systematic confusion won't fix by re-rolling)."""
    task, y = ep0["task"], ep0["y"]
    last_reason = "unattempted"
    wrong_guesses: list[str] = []
    for attempt in range(args.max_attempts):
        ep = ep0 if attempt < 2 else resample_episode(ep0, args, attempt)
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_episode_content(ep, args.descriptions_map)},
                ],
                # These deployments reject `max_tokens`.
                max_completion_tokens=args.max_completion_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface the class, keep going
            last_reason = f"api:{type(exc).__name__}"
            if attempt + 1 < args.max_attempts:
                backoff(attempt)
            continue

        good, reason, predicted = validate_episode(text, y)
        if good:
            return {
                "uid": task["uid"],
                "family": task["family"],
                "row_index": task["row_index"],
                "data_source": task["data_source"],
                "ground_truth": y,
                "model": args.model,
                "attempts": attempt + 1,
                "grounded": True,
                "response": text,
                # Ablation metadata (ignored by build_sft_parquet.py). Supports
                # keyed by plot filename -- uids collide across split halves.
                "candidates": ep["order"],
                "hard_negatives": ep["hard"],
                "supports": {
                    c: [Path(r["image_path"]).name for r in rows]
                    for c, rows in ep["supports"].items()
                },
                "n_shot": args.shots,
                "n_way": args.n_way,
                "resampled": attempt >= 2,
                "wrong_guesses": wrong_guesses,
                "descriptions": bool(args.descriptions_map),
                "seed": args.seed,
            }
        last_reason = reason
        if predicted is not None:
            wrong_guesses.append(predicted)
            with _print_lock:
                stats["confusion"][(y, predicted)] += 1

    with _print_lock:
        stats["dropped"] += 1
        stats["by_class"][y]["dropped"] += 1
        stats["drop_reasons"][last_reason] += 1
    return None


def dry_run(client, ep, args) -> None:
    print(f"query: {ep['task']['uid']} gt={ep['y']}")
    print(f"candidates (shuffled): {ep['order']}  hard={ep['hard']}")
    for cls in ep["order"]:
        print(f"  {cls}: {[Path(r['image_path']).name for r in ep['supports'][cls]]}")
    print("\n---- prompt layout ----")
    for part in build_episode_content(ep, args.descriptions_map):
        if part["type"] == "text":
            print(part["text"])
        else:
            print("[IMAGE]")
    print("\n---- live call ----")
    stats = _fresh_stats()
    rec = generate_one(client, ep, args, stats)
    if rec:
        print(json.dumps(rec, indent=2))
    else:
        print(f"FAILED: {dict(stats['drop_reasons'])}")


def _fresh_stats() -> dict:
    return {
        "kept": 0,
        "dropped": 0,
        "drop_reasons": collections.Counter(),
        "by_class": collections.defaultdict(lambda: {"kept": 0, "dropped": 0}),
        "confusion": collections.Counter(),
        "attempts": collections.Counter(),
    }


def report(stats: dict, n_todo: int, elapsed: float) -> None:
    print(
        f"\nkept={stats['kept']} dropped={stats['dropped']} "
        f"({stats['kept'] / max(1, n_todo):.1%} yield) in {elapsed:.0f}s"
    )
    if stats["drop_reasons"]:
        top = sorted(stats["drop_reasons"].items(), key=lambda kv: -kv[1])
        print(f"drop reasons: {dict(top)}")
    if stats["attempts"]:
        print(f"attempts histogram (kept): {dict(sorted(stats['attempts'].items()))}")
    by_class = stats["by_class"]
    if by_class:
        zero = sorted(c for c, s in by_class.items() if s["kept"] == 0)
        print(f"\nper-class yield ({len(by_class)} classes touched):")
        for cls in sorted(by_class):
            s = by_class[cls]
            print(f"  {cls:18s} kept={s['kept']} dropped={s['dropped']}")
        if zero:
            print(f"classes with ZERO kept: {zero}")
    if stats["confusion"]:
        print("\ntop confusions (gt -> predicted):")
        for (gt, pred), n in stats["confusion"].most_common(10):
            print(f"  {gt:18s} -> {pred:18s} x{n}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", type=Path, default=Path("data/sft/ts_sft_tasks.jsonl"))
    ap.add_argument(
        "--support-tasks", type=Path, default=Path("data/sft/rl_gallery_tasks.jsonl"),
        help="labelled support pool; the RL half, so every SFT row stays queryable",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=Path("data/sft/ts_images"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--shots", type=int, default=3, help="support plots per candidate")
    ap.add_argument("--hard-negatives", type=int, default=2, help="same-category negatives")
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--concurrency", type=int, default=4, help="~1-2 MB payload/request")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument(
        "--descriptions", type=Path, default=Path("data/sft/meta/text_description.json"),
        help="upstream SmellNet per-substance profiles, shown per candidate; "
        "missing file or --descriptions '' disables",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="one episode: layout + call + verdict")
    args = ap.parse_args()

    if args.hard_negatives > args.n_way - 1:
        raise SystemExit("--hard-negatives must be < --n-way")
    if args.n_way * args.shots + 1 > 50:
        raise SystemExit("n_way*shots+1 exceeds the endpoint's 50-image cap")

    args.descriptions_map = {}
    if str(args.descriptions) and args.descriptions.is_file():
        args.descriptions_map = json.loads(args.descriptions.read_text())
        print(f"substance profiles: {len(args.descriptions_map)} loaded")
    else:
        print("substance profiles: none (file missing or disabled)")

    queries = load_tasks(args.tasks, args.image_root)
    supports = load_tasks(args.support_tasks, args.image_root)
    args.pool_by_class = collections.defaultdict(list)
    for rec in supports:
        args.pool_by_class[rec["ground_truth"]].append(rec)
    missing = set(ALL_CLASSES) - set(args.pool_by_class)
    if missing:
        raise SystemExit(f"support pool missing classes: {sorted(missing)}")
    thin = {c: len(p) for c, p in args.pool_by_class.items() if len(p) < 2}
    if thin:
        print(f"WARNING: classes with <2 supports: {thin}")
    print(
        f"queries={len(queries)} supports={len(supports)} "
        f"({args.n_way}-way, {args.shots}-shot, {args.hard_negatives} hard)"
    )

    episodes = plan_episodes(queries, args)

    done = read_done_uids(args.out)
    todo = [e for e in episodes if e["task"]["uid"] not in done]
    # Shuffle so --limit samples classes evenly (uid order groups classes);
    # resume is unaffected -- skipping is keyed on uid, not position.
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"episodes={len(episodes)} already_done={len(done)} to_generate={len(todo)}")
    if not todo:
        print("nothing to do")
        return

    client = make_client(args.request_timeout)

    if args.dry_run:
        dry_run(client, todo[0], args)
        return

    stats = _fresh_stats()
    t0 = time.time()
    # Append + flush per record; as_completed so a slow call never holds back
    # finished results (see gen_sft_targets.py for the war stories).
    with args.out.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, e, args, stats) for e in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                with _print_lock:
                    stats["dropped"] += 1
                    if stats["dropped"] <= 3:
                        print(f"  worker error: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["kept"] += 1
            stats["by_class"][rec["ground_truth"]]["kept"] += 1
            stats["attempts"][rec["attempts"]] += 1
            if stats["kept"] % 10 == 0:
                rate = stats["kept"] / max(1e-9, time.time() - t0)
                print(f"  kept={stats['kept']} dropped={stats['dropped']} {rate:.2f}/s")

    report(stats, len(todo), time.time() - t0)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
