"""Answer-conditioned (STaR-style) SFT targets + shared API-client helpers.

GPT is given the question AND the correct answer and writes `<think>…</think>
\\boxed{answer}` in the exact format the GRPO reward scores. Safety rails:
resume-by-uid (never re-bills a row), validation via the real reward functions
from mirl_ext.rewards._common, and leak filters ("the correct answer is…").
Also home to the client/retry/resume helpers gen_sft_episodes imports.

    python mirl_ext/sft/gen_sft_targets.py --tasks sft_tasks.jsonl --out traces.jsonl
"""

from __future__ import annotations

import argparse
import base64
import collections
import functools
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

BASE_URL = "http://point.dd.works:18890/v1"

# Single model by default = one provenance per corpus; retries re-try the SAME
# model. --model-ladder opts into fallback. (gpt-5.3-chat is dead, 404.)
DEFAULT_MODEL = "gpt-5.6-sol_2026-07-09"
FALLBACK_MODELS = ["gpt-5.5_2026-04-24", "gpt-5.1_2025-11-13"]

KEY_PATHS = [
    Path.home() / ".config/mirl/microsoft_openai_key",
    Path.home() / "mit/rlm-compaction/.env",
]

# Phrases that reveal the answer was supplied.
_LEAK_RE = re.compile(
    r"\b(the (correct|given|provided|true|target) answer|we (are|were) told|"
    r"as (given|provided|stated above)|the label is|according to the (label|answer)|"
    r"since the answer)\b",
    re.IGNORECASE,
)

# Citing in-context material ("resembles the references") is a second leak: that
# context won't exist at rollout. Flag the WORDS (phrases are too fragile); bare
# "example" spared. gen_sft_episodes' _EPISODE_LEAK_RE subsumes this -- keep in sync.
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


def build_few_shot_turns(task, pool, k, image_root, with_images, seed):
    """In-context (question -> answer) demo turns: same data_source, distinct
    labels preferred, never the query row. Teaches label format + rendering style."""
    if not k:
        return []
    cands = [d for d in pool.get(task.get("data_source"), []) if d["uid"] != task["uid"]]
    if not cands:
        return []
    rng = random.Random(f"{seed}::{task['uid']}")
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
                content = [_img(str(path)), {"type": "text", "text": q}]
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


@functools.lru_cache(maxsize=512)
def _b64(path: str) -> str:
    # The same PNG recurs across calls (every demo turn; every episode support);
    # encode each file once.
    return base64.b64encode(Path(path).read_bytes()).decode()


def _img(path: str) -> dict:
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(path)}"}}


def read_done_uids(path: Path) -> set[str]:
    """Resume set: uids already in the output JSONL (tolerates a truncated tail)."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def make_client(timeout: float):
    """Explicit timeout, no SDK retries (defaults can stall a worker ~30 min);
    the caller's --max-attempts loop owns the retry policy."""
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=load_api_key(), timeout=timeout, max_retries=0)


def backoff(attempt: int) -> None:
    """Exponential + jitter (no thundering herd). Callers skip it after the final attempt."""
    time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))


def build_blind_user_prompt(task: dict) -> str:
    """Question only, answer withheld -- blind accuracy bounds what RL can reach."""
    question = task["prompt"].replace("<image>", "").replace("<video>", "").strip()
    return f"QUESTION:\n{question}\n\nAnswer from the attached media."


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


def _fresh_stats() -> dict:
    return {"kept": 0, "dropped": 0, "drop_reasons": collections.Counter()}


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
    question = task["prompt"].replace("<image>", "").replace("<video>", "").strip()
    return (
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER:\n{task['ground_truth']}\n\n"
        "Write the completion now."
    )


def build_user_content(task: dict, image_root: Path | None, grounded: bool, blind: bool = False):
    """(content, used_image): text alone, or plot image + text. Missing files fall
    back to text-only rather than failing the row."""
    text = build_blind_user_prompt(task) if blind else build_user_prompt(task)
    if not grounded:
        return text, False
    frames = task.get("frame_paths") or []
    if frames:  # video task: staged frames (see stage_media.py), oldest -> newest
        resolved = [_resolve_image(f, image_root) for f in frames]
        parts = [_img(str(r)) for r in resolved if r is not None]
        if parts:
            return parts + [{"type": "text", "text": text}], True
        return text, False
    raw = task.get("image_path") or ""
    if not raw:
        return text, False
    path = _resolve_image(raw, image_root)
    if path is None:
        return text, False
    return [_img(str(path)), {"type": "text", "text": text}], True


def _norm(s: str) -> str:
    # Generic conservative equality gate -- stricter than or equal to every family
    # reward normalizer (rewards/ecg.py _norm is byte-identical; smellnet's
    # _norm_label is coarser), so acceptance never over-approximates the reward.
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
    # Deterministic per task (rng keyed seed::uid) -- build once, retry many.
    content, used_image = build_user_content(
        task, args.image_root, args.grounded, blind=args.blind
    )
    if args.blind:
        system = BLIND_SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT + (GROUNDED_SUFFIX if used_image else "")
    demo_turns = build_few_shot_turns(
        task, args.demo_pool, args.few_shot, args.image_root,
        args.few_shot_images, args.seed,
    )

    last_reason = "unattempted"
    ladder = args.ladder
    for attempt in range(args.max_attempts):
        model = ladder[min(attempt, len(ladder) - 1)]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}]
                + demo_turns
                + [{"role": "user", "content": content}],
                # These deployments reject `max_tokens`; the parameter is
                # `max_completion_tokens`. Using the wrong one 400s every call.
                max_completion_tokens=args.max_completion_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface the class, keep going
            last_reason = f"api:{type(exc).__name__}"
            if attempt + 1 < args.max_attempts:
                backoff(attempt)
            continue

        record = {
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
        if args.blind:
            # Format-only gate: rejecting wrong answers would inflate accuracy to ~100%.
            boxed = extract_boxed_answer(text) if text else None
            if boxed is not None and format_reward(text) == 1.0:
                return {**record, "predicted": boxed, "n_shot": len(demo_turns) // 2}
            last_reason = "format" if text else "empty"
            continue

        good, reason = validate(text, task["ground_truth"])
        if good:
            return record
        last_reason = reason

    with _print_lock:
        stats["dropped"] += 1
        stats["drop_reasons"][last_reason] += 1
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
        "--data-sources",
        nargs="*",
        default=None,
        help="restrict to these data_source values (e.g. smellnet_base). Needed because "
        "smellnet base and mixture are different tasks on different hardware and pooling "
        "them makes either number uninterpretable.",
    )
    ap.add_argument("--few-shot", type=int, default=0, help="in-context demos per call")
    ap.add_argument(
        "--few-shot-pool",
        default=None,
        help="JSONL of grounded traces to draw demos from (required with --few-shot)",
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

    all_tasks = [
        json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()
    ]

    # Demo pool: grounded traces joined to their task record, grouped by data_source.
    # Demos MUST come from the SFT half (the traces already do) so nothing from the RL
    # half leaks into an in-context example. Keyed on the UNFILTERED task list so
    # demos stay available even when --families/--data-sources restricts the queries.
    args.demo_pool = {}
    if args.few_shot:
        if not args.few_shot_pool:
            raise SystemExit("--few-shot requires --few-shot-pool (grounded traces JSONL)")
        by_uid = {t["uid"]: t for t in all_tasks}
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

    tasks = all_tasks
    if args.families:
        tasks = [t for t in tasks if t["family"] in args.families]
    if args.data_sources:
        tasks = [t for t in tasks if t.get("data_source") in set(args.data_sources)]

    out_path = Path(args.out)
    done = read_done_uids(out_path)
    todo = [t for t in tasks if t["uid"] not in done]
    # Shuffle so --limit samples all families; resume is keyed on uid, not position.
    if args.shuffle:
        random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]

    print(f"tasks={len(tasks)} already_done={len(done)} to_generate={len(todo)}")
    if not todo:
        print("nothing to do")
        return

    client = make_client(args.request_timeout)

    if args.dry_run:
        stats = _fresh_stats()
        rec = generate_one(client, todo[0], args, stats)
        print(json.dumps(rec, indent=2) if rec else f"FAILED: {dict(stats['drop_reasons'])}")
        return

    stats = _fresh_stats()
    t0 = time.time()
    # Append+flush per record (a kill must not lose paid work); as_completed, NOT
    # pool.map (input-order yields let one slow call stall the resume checkpoint).
    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, t, args, stats) for t in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                stats["dropped"] += 1
                rec = None
                if stats["dropped"] <= 3:
                    print(f"  worker error: {type(exc).__name__}: {str(exc)[:120]}")
            if rec is None:
                continue
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
    if stats["drop_reasons"]:
        print(f"drop reasons: {dict(stats['drop_reasons'].most_common())}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
