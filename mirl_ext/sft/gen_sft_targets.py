"""Prediction-based SFT traces + the shared API-client helpers.

GPT answers each task from the question + attached media ALONE (never shown the
answer) in `<think>…</think> \\boxed{answer}` form; a trace is kept iff the boxed
answer matches ground truth. Yield therefore IS the teacher's accuracy, and every
kept trace is earned. For classification/MCQ families (climb, tactile, ecg);
smellnet needs support examples -> gen_sft_episodes. Resume-by-uid: reruns skip
finished rows. gen_sft_episodes imports the client helpers from here.

    python mirl_ext/sft/gen_sft_targets.py --tasks tasks.jsonl --out traces.jsonl --grounded
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.rewards._common import extract_boxed_answer, format_reward  # noqa: E402

BASE_URL = "http://point.dd.works:18890/v1"
# TRAPI needs the full dated deployment id (bare "gpt-5.6-sol" 404s).
DEFAULT_MODEL = "gpt-5.6-sol_2026-07-09"

KEY_PATHS = [
    Path.home() / ".config/mirl/microsoft_openai_key",
    Path.home() / "mit/rlm-compaction/.env",
]

# _LEAK_RE guards traces written with the answer visible; kept for import by
# gen_sft_episodes (its prompts state rules that mention "the answer").
_LEAK_RE = re.compile(
    r"\b(the (correct|given|provided|true|target) answer|we (are|were) told|"
    r"as (given|provided|stated above)|the label is|according to the (label|answer)|"
    r"since the answer)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are an expert at reading medical images, sensor recordings, and "
    "interaction videos.\n"
    "Answer the question from the attached media ALONE. You are NOT given the answer.\n\n"
    "Rules:\n"
    "1. Output EXACTLY this shape and nothing else:\n"
    "   <think> reasoning </think> \\boxed{answer}\n"
    "2. Base the reasoning on features you can actually see -- specific "
    "channels/regions, where in time features occur, relative amplitudes.\n"
    "3. If the question lists options, \\boxed{} MUST hold exactly one of them "
    "verbatim (or comma-separated letters, in order, for select-all-that-apply).\n"
    "4. Keep the reasoning to 2-4 sentences. Concise beats florid.\n"
)


# ---- shared helpers (imported by gen_sft_episodes) ----

def load_api_key() -> str:
    """Read the key from disk/env. Never printed, never logged."""
    if os.environ.get("MIRL_OPENAI_KEY"):
        return os.environ["MIRL_OPENAI_KEY"].strip()
    for path in KEY_PATHS:
        if not path.is_file():
            continue
        text = path.read_text()
        if path.name.endswith(".env"):
            for line in text.splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
        elif text.strip():
            return text.strip()
    raise SystemExit("No API key. Put it in ~/.config/mirl/microsoft_openai_key or MIRL_OPENAI_KEY.")


def make_client(timeout: float):
    """Explicit timeout, no SDK retries -- the caller's attempt loop owns retry policy."""
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=load_api_key(), timeout=timeout, max_retries=0)


def backoff(attempt: int) -> None:
    """Exponential + jitter. Callers skip it after the final attempt."""
    time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))


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


def _resolve_image(raw, image_root):
    path = Path(raw)
    if image_root is not None:
        cand = image_root / path.parent.name / path.name
        if cand.is_file():
            return cand
    return path if path.is_file() else None


@functools.lru_cache(maxsize=512)
def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _img(path: str) -> dict:
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(path)}"}}


def _norm(s: str) -> str:
    # Conservative equality gate -- stricter than or equal to every family reward.
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ---- prediction-based generation ----

def build_user_content(task: dict, image_root, grounded: bool):
    """(content, used_image): the question, plus the media if grounded. No answer."""
    question = task["prompt"].replace("<image>", "").replace("<video>", "").strip()
    text = f"QUESTION:\n{question}\n\nAnswer from the attached media."
    if not grounded:
        return text, False
    frames = task.get("frame_paths") or []
    if frames:  # video task: staged frames, oldest -> newest (see stage_media.py)
        resolved = [_resolve_image(f, image_root) for f in frames]
        parts = [_img(str(r)) for r in resolved if r is not None]
        if parts:
            return parts + [{"type": "text", "text": text}], True
        return text, False
    if task.get("image_path"):
        path = _resolve_image(task["image_path"], image_root)
        if path is not None:
            return [_img(str(path)), {"type": "text", "text": text}], True
    return text, False


def validate(text: str, ground_truth: str) -> tuple[bool, str, str | None]:
    """(accepted, reason, predicted): reward-format gates + boxed == ground truth."""
    if not text:
        return False, "empty", None
    if format_reward(text) != 1.0:
        return False, "format", None
    boxed = extract_boxed_answer(text)
    if boxed is None:
        return False, "no_boxed", None
    if _norm(boxed) != _norm(ground_truth):
        return False, "wrong", _norm(boxed)
    return True, "ok", _norm(boxed)


def generate_one(client, task: dict, args, stats) -> dict | None:
    content, used_image = build_user_content(task, args.image_root, args.grounded)
    last_reason = "unattempted"
    wrong_guesses: list[str] = []
    for attempt in range(args.max_attempts):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                # This endpoint rejects `max_tokens`.
                max_completion_tokens=args.max_completion_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_reason = f"api:{type(exc).__name__}"
            if attempt + 1 < args.max_attempts:
                backoff(attempt)
            continue
        good, reason, predicted = validate(text, task["ground_truth"])
        if good:
            return {
                "uid": task["uid"],
                "family": task["family"],
                "row_index": task["row_index"],
                "data_source": task.get("data_source"),
                "ground_truth": task["ground_truth"],
                "model": args.model,
                "attempts": attempt + 1,
                "grounded": used_image,
                "wrong_guesses": wrong_guesses,
                "response": text,
            }
        last_reason = reason
        if predicted is not None:
            wrong_guesses.append(predicted)
    stats[last_reason] += 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--grounded", action="store_true", help="attach the recording's media")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument(
        "--skip-sources", nargs="*", default=["description"],
        help="data_sources to exclude (default: tactile's open-response captions, "
        "which exact-match can't score)",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="one call, print result, exit")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.families:
        tasks = [t for t in tasks if t["family"] in args.families]
    if args.skip_sources:
        tasks = [t for t in tasks if t.get("data_source") not in set(args.skip_sources)]
    done = read_done_uids(args.out)
    todo = [t for t in tasks if t["uid"] not in done]
    # Shuffle so --limit samples all families; resume is keyed on uid, not position.
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"tasks={len(tasks)} already_done={len(done)} to_generate={len(todo)}")
    if not todo:
        return

    client = make_client(args.request_timeout)
    stats: collections.Counter = collections.Counter()

    if args.dry_run:
        rec = generate_one(client, todo[0], args, stats)
        print(json.dumps(rec, indent=2) if rec else f"FAILED: {dict(stats)}")
        return

    kept, t0 = 0, time.time()
    # Append+flush per record (a kill must not lose paid work); as_completed so a
    # slow call never stalls the resume checkpoint.
    with args.out.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(generate_one, client, t, args, stats) for t in todo]):
            rec = fut.result()
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            kept += 1
            if kept % 25 == 0:
                print(f"  kept={kept} dropped={sum(stats.values())} {kept / (time.time() - t0):.1f}/s")

    print(f"\nkept={kept} dropped={sum(stats.values())} ({kept / max(1, len(todo)):.1%} yield)")
    if stats:
        print(f"drop reasons: {dict(stats.most_common())}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
