# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Generate `<think>…</think> \\boxed{answer}` SFT targets with the GPT-5.x endpoint.

Answer-conditioned rationalization (STaR-style): GPT is given the question and the
CORRECT answer and writes a completion in the exact format the GRPO reward scores.
It never sees the signal, so the reasoning is plausible-but-ungrounded; that is the
accepted trade for a cold start, and RL grounds it afterwards.

Three things make this safe to run against a metered endpoint:

* **Resume.** Completions append to JSONL and every already-finished uid is skipped
  on restart, so an interrupted run never re-bills a row.
* **Validation before acceptance.** Each completion must satisfy the *real* reward
  regex imported from ``mirl_ext.rewards._common`` -- not a local copy that could
  drift -- and its ``\\boxed{}`` must equal the ground truth. Non-conforming
  completions are retried, then dropped and counted.
* **Leak filter.** A rationalization that says "the correct answer is X" or "we are
  told…" teaches the student to cite an oracle that will not exist at inference.
  Those are rejected even though they pass the format regex.

    # 1. on the cluster
    python scripts/mirl/export_sft_tasks.py --limit-per-family 2500
    # 2. anywhere with internet
    python scripts/mirl/gen_sft_targets.py --tasks sft_tasks.jsonl --out traces.jsonl \\
        --limit 50 --concurrency 8
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
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
from gallery import (  # noqa: E402
    CONTRASTIVE_CLAUSE,
    GALLERY_INTRO,
    build_gallery,
    descriptions_block,
    load_descriptions,
)

BASE_URL = "http://point.dd.works:18890/v1"

# Single model by default so one generation run yields ONE provenance, not a silent
# mix: an earlier run fell back and produced 341 gpt-5.5 rows among 1087 gpt-5.6 ones,
# which makes the corpus impossible to attribute. Retries now re-try the SAME model.
# Pass --model-ladder to opt back into cross-model fallback.
# (gpt-5.6-sol verified live 2026-07-30. gpt-5.3-chat_2026-03-03 is dead: 404
# DEPLOYMENT_NOT_FOUND, verified 2026-07-26.)
DEFAULT_MODEL = "gpt-5.6-sol_2026-07-09"
FALLBACK_MODELS = ["gpt-5.5_2026-04-24", "gpt-5.1_2025-11-13"]

KEY_PATHS = [
    Path.home() / ".config/mirl/microsoft_openai_key",
    Path.home() / "mit/rlm-compaction/.env",
]

# Phrases that reveal the answer was supplied. The student would learn to emit them
# and then have nothing to fall back on at inference, where no answer is given.
_LEAK_RE = re.compile(
    r"\b(the (correct|given|provided|true|target) answer|we (are|were) told|"
    r"as (given|provided|stated above)|the label is|according to the (label|answer)|"
    r"since the answer)\b",
    re.IGNORECASE,
)

# Gallery citations are a SECOND kind of leak, and the more dangerous one because it
# does not look like cheating. A judge panel reading build_sft_parquet.py found that
# the SFT prompt carries no gallery at all -- it is present only while the trace is
# WRITTEN -- so "resembles the conduction-disturbance references" trains the student to
# cite context that will never exist at rollout time. It is also circular: you can only
# assert resemblance to the correct class's examples because you were handed the answer.
# Requesting this in the prompt is not enough; reject it, so a violation costs a retry
# rather than silently entering the corpus.
# Matching the citing PHRASE is too fragile -- "resembles the conduction-disturbance
# references" defeats a \w* because of the hyphen, and there are unlimited paraphrases.
# The prompt forbids these words outright, so flag the WORDS. Plural "examples" is the
# citing form; bare "example" is spared so an ordinary "for example" is not punished.
_GALLERY_LEAK_RE = re.compile(
    r"\b(galler(y|ies)|references?|examples)\b",
    re.IGNORECASE,
)

BLIND_SYSTEM_PROMPT = (
    "You are an expert at reading scientific plots of sensor and physiological signals.\n"
    "Answer the question from the attached plot ALONE. You are NOT given the answer.\n\n"
    "Rules:\n"
    "1. Output EXACTLY this shape and nothing else:\n"
    "   <think> reasoning </think> \\boxed{answer}\n"
    "2. Base the reasoning on features you can actually see in the plot.\n"
    "3. If the question lists options, \\boxed{} MUST hold exactly one of them, verbatim.\n"
    "4. Keep the reasoning to 2-4 sentences.\n"
)


def build_gallery_system_content(system_text, family, cache):
    """Fold the gallery INTO the system message: [text, image, image, ...].

    Verified accepted by this endpoint. Keeping the gallery in its own user turn is an
    extra degree of freedom with no justification behind it, so this exists to A/B the
    placement on matched queries rather than to argue about it.
    """
    entry = cache.get(family)
    if not entry:
        return system_text
    sheets, note = entry["sheets"], entry["note"]
    content = [{"type": "text", "text": system_text + "\n\n" + GALLERY_INTRO.format(note=note)}]
    for sheet in sheets:
        b64 = base64.b64encode(Path(sheet).read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    return content


def build_gallery_turns(family, cache):
    """Leading (gallery -> ack) turns for a family. Built once, reused every call."""
    entry = cache.get(family)
    if not entry:
        return []
    sheets, note = entry["sheets"], entry["note"]
    content = [{"type": "text", "text": GALLERY_INTRO.format(note=note)}]
    for sheet in sheets:
        b64 = base64.b64encode(Path(sheet).read_bytes()).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Gallery reviewed. Ready for the query."},
    ]


def build_few_shot_turns(task, pool, k, image_root, with_images, seed):
    """Prior (question -> answer) turns from OTHER rows, as in-context demonstrations.

    Zero-shot is the wrong measure for these tasks, and smellnet shows why: its
    mixture prompt lists NO options and only hints the label shape with one example
    ("e.g. 'Almond20_Orange80'"), while scoring is exact string match over 86 recipes
    written in three different conventions. A model cannot infer that from nothing.
    Demonstrations supply both the label FORMAT and what each class looks like in this
    specific rendering style, which is the thing zero-shot cannot know.

    Demos are drawn from the same data_source, prefer DISTINCT labels (so the model
    sees range rather than one class k times), and never include the query row.
    """
    import random as _r

    if not k:
        return []
    cands = [d for d in pool.get(task.get("data_source"), []) if d["uid"] != task["uid"]]
    if not cands:
        return []
    rng = _r.Random(f"{seed}::{task['uid']}")
    rng.shuffle(cands)
    picked, seen = [], set()
    for d in cands:                       # first pass: one per distinct label
        if d["ground_truth"] not in seen:
            picked.append(d)
            seen.add(d["ground_truth"])
        if len(picked) == k:
            break
    for d in cands:                       # top up if the family has few labels
        if len(picked) == k:
            break
        if d not in picked:
            picked.append(d)

    turns = []
    for d in picked:
        q = build_blind_user_prompt(d)
        content = q
        if with_images and d.get("image_path"):
            path = _resolve_image(d["image_path"], image_root)
            if path is not None:
                b64 = base64.b64encode(path.read_bytes()).decode()
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": q},
                ]
        turns.append({"role": "user", "content": content})
        turns.append({"role": "assistant", "content": d["response"]})
    return turns


def _resolve_image(raw, image_root):
    path = Path(raw)
    if image_root is not None:
        cand = image_root / path.parent.name / path.name
        if cand.is_file():
            return cand
    return path if path.is_file() else None


def build_blind_user_prompt(task: dict) -> str:
    """Question only -- the answer is withheld.

    This is the honest test of whether the plot carries the signal: an
    answer-conditioned trace scores 100% by construction, so it measures nothing
    about readability. Blind accuracy bounds what RL can realistically reach.
    """
    question = task["prompt"].replace("<image>", "").strip()
    return f"QUESTION:\n{question}\n\nAnswer from the plot."


GROUNDED_SUFFIX = (
    "\n6. An image of the actual recording is attached. Your reasoning MUST describe "
    "what is genuinely visible in THAT plot -- specific leads/channels, where in time "
    "features occur, relative amplitudes. Do not emit generic findings that would fit "
    "any recording with this label.\n"
)

SYSTEM_PROMPT = (
    "You write supervised fine-tuning targets for a multimodal signal-reasoning model.\n"
    "You are given a question and its correct answer. Write the completion the model "
    "should have produced.\n\n"
    "Rules:\n"
    "1. Output EXACTLY this shape and nothing else:\n"
    "   <think> reasoning </think> \\boxed{answer}\n"
    "2. The text inside \\boxed{} must be the correct answer, copied VERBATIM.\n"
    "3. The reasoning must read as if you derived the answer from the signal yourself. "
    "NEVER mention that the answer was given to you. Do not write 'the correct answer', "
    "'we are told', 'as provided', or anything similar.\n"
    "4. Reference concrete, modality-appropriate evidence (waveform morphology, sensor "
    "response shape, pressure distribution, image findings) that would plausibly support "
    "this answer.\n"
    "5. Keep the reasoning to 2-4 sentences. Concise beats florid.\n"
)

_print_lock = threading.Lock()


def load_api_key() -> str:
    """Read the key from disk/env. Never printed, never logged, never echoed."""
    if os.environ.get("MIRL_OPENAI_KEY"):
        return os.environ["MIRL_OPENAI_KEY"].strip()
    for path in KEY_PATHS:
        if not path.is_file():
            continue
        text = path.read_text()
        if path.suffix == ".env" or path.name == ".env":
            for line in text.splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
        else:
            key = text.strip()
            if key:
                return key
    raise SystemExit(
        "No API key found. Put it in ~/.config/mirl/microsoft_openai_key "
        "or set MIRL_OPENAI_KEY."
    )


def build_user_prompt(task: dict) -> str:
    question = task["prompt"].replace("<image>", "").strip()
    return (
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER:\n{task['ground_truth']}\n\n"
        "Write the completion now."
    )


def build_user_content(task: dict, image_root: Path | None, grounded: bool, blind: bool = False):
    """User turn: text alone, or the plot image plus text.

    Attaching the plot is what turns the `<think>` from a label-conditioned template
    into a description of THIS recording. Measured on the text-only run: ECG traces
    for the same label reached 0.71 mean pairwise Jaccard with 7 byte-identical
    duplicates among 29 "Normal" rows -- i.e. different patients received the same
    invented findings. The plots are readable (8 stacked lead traces, per-channel
    e-nose curves, a taxel heatmap), so vision has real signal to describe.

    Returns (content, used_image). Falls back to text-only when the file is missing
    rather than failing the row, and reports it so partial grounding is visible.
    """
    text = build_blind_user_prompt(task) if blind else build_user_prompt(task)
    if not grounded:
        return text, False
    raw = task.get("image_path") or ""
    if not raw:
        return text, False
    path = _resolve_image(raw, image_root)
    if path is None:
        return text, False
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text},
        ],
        True,
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def validate(text: str, ground_truth: str) -> tuple[bool, str]:
    """Accept only what the GRPO reward would score as a well-formed, correct answer."""
    if not text:
        return False, "empty"
    if format_reward(text) != 1.0:
        return False, "format"
    boxed = extract_boxed_answer(text)
    if boxed is None:
        return False, "no_boxed"
    if _norm(boxed) != _norm(ground_truth):
        return False, "answer_mismatch"
    if _LEAK_RE.search(text):
        return False, "oracle_leak"
    if _GALLERY_LEAK_RE.search(text):
        return False, "gallery_leak"
    return True, "ok"


def generate_one(client, task: dict, args, stats: dict) -> dict | None:
    """One task -> one validated completion, or None. Retries across the ladder."""
    last_reason = "unattempted"
    ladder = args.ladder
    for attempt in range(args.max_attempts):
        model = ladder[min(attempt, len(ladder) - 1)]
        try:
            content, used_image = build_user_content(
                task, args.image_root, args.grounded, blind=args.blind
            )
            if args.blind:
                system = BLIND_SYSTEM_PROMPT
            else:
                system = SYSTEM_PROMPT + (GROUNDED_SUFFIX if used_image else "")
            demo_turns = (
                [] if args.gallery_in_system
                else build_gallery_turns(task["family"], args.gallery_cache)
            )
            demo_turns += build_few_shot_turns(
                task, args.demo_pool, args.few_shot, args.image_root,
                args.few_shot_images, args.seed,
            )
            if args.gallery_cache.get(task["family"]):
                system = system + CONTRASTIVE_CLAUSE
            if args.desc_block:
                system = system + "\n" + args.desc_block
            sys_content = (
                build_gallery_system_content(system, task["family"], args.gallery_cache)
                if args.gallery_in_system else system
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_content}]
                + demo_turns
                + [{"role": "user", "content": content}],
                # These deployments reject `max_tokens`; the parameter is
                # `max_completion_tokens`. Using the wrong one 400s every call.
                max_completion_tokens=args.max_completion_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface the class, keep going
            last_reason = f"api:{type(exc).__name__}"
            # Exponential backoff with jitter so a rate limit does not resynchronize
            # every worker into a thundering herd on the next tick.
            time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
            continue

        if args.blind:
            # Only the FORMAT is required. Rejecting wrong answers here would filter
            # the sample down to the cases the model got right and report ~100%.
            boxed = extract_boxed_answer(text) if text else None
            if boxed is not None and format_reward(text) == 1.0:
                return {
                    "uid": task["uid"],
                    "family": task["family"],
                    "row_index": task["row_index"],
                    "data_source": task.get("data_source"),
                    "ground_truth": task["ground_truth"],
                    "predicted": boxed,
                    "model": model,
                    "attempts": attempt + 1,
                    "grounded": used_image,
                    "n_shot": len(demo_turns) // 2,
                    "response": text,
                }
            last_reason = "format" if text else "empty"
            continue

        good, reason = validate(text, task["ground_truth"])
        if good:
            return {
                "uid": task["uid"],
                "family": task["family"],
                "row_index": task["row_index"],
                "data_source": task.get("data_source"),
                "ground_truth": task["ground_truth"],
                "model": model,
                "attempts": attempt + 1,
                "grounded": used_image,
                "response": text,
            }
        last_reason = reason

    with _print_lock:
        stats["dropped"] += 1
        stats.setdefault("drop_reasons", {})
        stats["drop_reasons"][last_reason] = stats["drop_reasons"].get(last_reason, 0) + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", required=True, help="JSONL from export_sft_tasks.py")
    ap.add_argument("--out", required=True, help="JSONL of validated completions (appended)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (SDK default 600 with 2 retries can hang "
        "a worker for ~30 min)",
    )
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--gallery",
        type=int,
        default=0,
        metavar="PER_CLASS",
        help="prepend a labelled reference gallery with N examples per class "
        "(contact sheets; beats the 50-image/request cap)",
    )
    ap.add_argument("--gallery-grid", type=int, default=3, help="sheet is GRID x GRID panels")
    ap.add_argument(
        "--gallery-in-system",
        action="store_true",
        help="put the gallery images in the SYSTEM message instead of a separate user turn",
    )
    ap.add_argument(
        "--gallery-tasks",
        nargs="*",
        default=None,
        help="draw gallery REFERENCES from this task file instead of --tasks. Point it "
        "at the RL half so every SFT-half row stays available as a query; base has only "
        "2-3 rows per class per half, so sharing one pool caps the gallery at 1/class.",
    )
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="SmellNet text_description.json -- per-substance volatile chemistry and "
        "expected sensor response, injected as system-prompt priors",
    )
    ap.add_argument(
        "--descriptions-mode", default="compact", choices=["compact", "full", "none"]
    )
    ap.add_argument(
        "--no-attach-gallery",
        action="store_true",
        help="build the gallery for its EXCLUSION set but do not send it. This is the "
        "matched control arm: both conditions then answer the identical query rows, so "
        "a difference cannot come from one arm having easier queries.",
    )
    ap.add_argument(
        "--data-sources",
        nargs="*",
        default=None,
        help="restrict to these data_source values (e.g. smellnet_base). Needed because "
        "smellnet base and mixture are different tasks on different hardware and pooling "
        "them makes either number uninterpretable.",
    )
    ap.add_argument(
        "--gallery-cache-dir", type=Path, default=Path("data/sft/gallery_cache")
    )
    ap.add_argument("--few-shot", type=int, default=0, help="in-context demos per call")
    ap.add_argument(
        "--few-shot-pool",
        default=None,
        help="JSONL of grounded traces to draw demos from (default: --tasks' traces)",
    )
    ap.add_argument(
        "--no-few-shot-images",
        dest="few_shot_images",
        action="store_false",
        help="text-only demos (teaches label FORMAT but not plot appearance; far cheaper)",
    )
    ap.add_argument(
        "--blind",
        action="store_true",
        help="withhold the answer and record the model's PREDICTION (eval, not SFT). "
        "Only the output FORMAT is validated -- rejecting wrong answers would filter "
        "the sample to the ones it got right and report ~100%%.",
    )
    ap.add_argument(
        "--grounded",
        action="store_true",
        help="attach the recording's plot PNG so reasoning describes the real signal",
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="local dir holding <family>/<hash>.png when plots were staged off-cluster",
    )
    ap.add_argument(
        "--model-ladder",
        action="store_true",
        help="on retry, fall back to older models (mixes provenance; off by default)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="keep task-file order (default: shuffle so --limit samples all families)",
    )
    ap.add_argument("--dry-run", action="store_true", help="one call, print result, exit")
    args = ap.parse_args()
    args.ladder = [args.model] + (FALLBACK_MODELS if args.model_ladder else [])

    # Demo pool: grounded traces joined to their task record, grouped by data_source.
    # Demos MUST come from the SFT half (the traces already do) so nothing from the RL
    # half leaks into an in-context example.
    # ---- reference galleries, built once per family ----
    args.gallery_cache = {}
    if args.gallery:
        gal_files = args.gallery_tasks or [args.tasks]
        all_tasks = []
        for pri, gf in enumerate(gal_files):
            for line in Path(gf).read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec["_pri"] = pri        # earlier file = preferred source
                    all_tasks.append(rec)
        if args.gallery_tasks:
            print(f"gallery reference pools (in preference order): {gal_files}")
        ref_keys: set[str] = set()
        for fam in sorted({t["family"] for t in all_tasks}):
            sheets, refs, note = build_gallery(
                all_tasks, fam, args.gallery, args.gallery_cache_dir,
                args.image_root, grid=args.gallery_grid,
                allowed_sources=set(args.data_sources) if args.data_sources else None,
            )
            if not sheets:
                continue
            if not args.no_attach_gallery:
                args.gallery_cache[fam] = {"sheets": sheets, "note": note}
            # Key the exclusion on image_path, NOT uid. uid is "<family>#<row_index>"
            # and row_index restarts at 0 in each split half, so 984 RL-half gallery
            # uids collide with UNRELATED SFT query rows -- which silently dropped 94
            # of 121 smellnet base queries. The plot path is unique across halves.
            ref_keys |= {r["image_path"] for r in refs if r.get("image_path")}
            n_by_pool = collections.Counter(r.get("_pri", 0) for r in refs)
            print(
                f"gallery[{fam}]: {len(sheets)} sheet(s) from {len(refs)} reference rows "
                f"(by pool: {dict(sorted(n_by_pool.items()))})"
            )
        # A gallery row used as its own query is trivially correct; drop them.
        args.gallery_ref_keys = ref_keys
    else:
        args.gallery_ref_keys = set()

    args.demo_pool = {}
    if args.few_shot:
        if not args.few_shot_pool:
            raise SystemExit("--few-shot requires --few-shot-pool (grounded traces JSONL)")
        by_uid = {
            json.loads(l)["uid"]: json.loads(l)
            for l in Path(args.tasks).read_text().splitlines()
            if l.strip()
        }
        n = 0
        for line in Path(args.few_shot_pool).read_text().splitlines():
            if not line.strip():
                continue
            tr = json.loads(line)
            src = by_uid.get(tr["uid"])
            if src is None:
                continue
            args.demo_pool.setdefault(src.get("data_source"), []).append(
                {**src, "response": tr["response"]}
            )
            n += 1
        print(
            f"demo pool: {n} traces over {len(args.demo_pool)} data_sources "
            f"({args.few_shot}-shot, images={args.few_shot_images})"
        )

    from openai import OpenAI

    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    if args.families:
        tasks = [t for t in tasks if t["family"] in args.families]
    if args.data_sources:
        tasks = [t for t in tasks if t.get("data_source") in set(args.data_sources)]

    # ---- substance priors ----
    args.desc_block = ""
    if args.descriptions:
        descs = load_descriptions(args.descriptions, args.descriptions_mode)
        labels = sorted({t["ground_truth"] for t in tasks if t["ground_truth"] in descs})
        args.desc_block = descriptions_block(descs, labels)
        if args.desc_block:
            print(
                f"substance priors: {len(labels)} labels, ~{len(args.desc_block)//4:,} tokens "
                f"(mode={args.descriptions_mode})"
            )


    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    n_before = len([t for t in tasks if t["uid"] not in done])
    todo = [
        t for t in tasks
        if t["uid"] not in done and t.get("image_path") not in args.gallery_ref_keys
    ]
    if args.gallery_ref_keys:
        print(
            f"excluded {n_before - len(todo)} query rows that are gallery references "
            f"(gallery pool holds {len(args.gallery_ref_keys)} plots)"
        )
    # The task file is grouped by family, so a bare head-slice under --limit would be
    # 100% smellnet and its yield would say nothing about the other five. Shuffle on a
    # fixed seed: representative sample, still reproducible, and resume is unaffected
    # because skipping is keyed on uid rather than position.
    if args.shuffle:
        random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"tasks={len(tasks)} already_done={len(done)} to_generate={len(todo)}")
    if not todo:
        print("nothing to do")
        return

    # Explicit timeout + no SDK-level retries. The default is a 600s timeout with 2
    # retries, so ONE unlucky call can block for ~30 minutes. Combined with the ordered
    # writer below that produced a silent 25-minute stall in which both runs looked
    # hung while the endpoint was healthy. We already retry ourselves (--max-attempts)
    # with backoff, so let the SDK fail fast and let our loop own the retry policy.
    client = OpenAI(
        base_url=BASE_URL,
        api_key=load_api_key(),
        timeout=args.request_timeout,
        max_retries=0,
    )

    if args.dry_run:
        stats: dict = {"dropped": 0}
        rec = generate_one(client, todo[0], args, stats)
        print(json.dumps(rec, indent=2) if rec else f"FAILED: {stats}")
        return

    stats = {"dropped": 0, "kept": 0}
    write_lock = threading.Lock()
    t0 = time.time()
    # Append + flush per record: a kill mid-run must not lose completed work, since
    # every lost line is a row that gets billed twice.
    # as_completed, NOT pool.map: map yields results in INPUT order, so a single slow
    # call holds back every finished result behind it. Writes then arrive in bursts and
    # a stall is indistinguishable from slowness -- and worse, the resume checkpoint
    # falls arbitrarily far behind the work actually done.
    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, t, args, stats) for t in todo]
        done_n = 0
        for fut in as_completed(futures):
            done_n += 1
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                stats["dropped"] += 1
                rec = None
                if stats["dropped"] <= 3:
                    print(f"  worker error: {type(exc).__name__}: {str(exc)[:120]}")
            if rec is None:
                continue
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                stats["kept"] += 1
                if stats["kept"] % 25 == 0:
                    rate = stats["kept"] / max(1e-9, time.time() - t0)
                    print(f"  kept={stats['kept']} dropped={stats['dropped']} {rate:.1f}/s")

    elapsed = time.time() - t0
    print(
        f"\nkept={stats['kept']} dropped={stats['dropped']} "
        f"({stats['kept'] / max(1, len(todo)):.1%} yield) in {elapsed:.0f}s"
    )
    if stats.get("drop_reasons"):
        print(f"drop reasons: {stats['drop_reasons']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
