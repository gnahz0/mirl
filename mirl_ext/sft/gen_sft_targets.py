"""Answer-blind zero-shot SFT trace generation (+ the shared API-client helpers).

For every exported task the teacher sees the fixed family context
(teacher_context.py), the task's own label definitions, the original question,
and the query media -- never the ground truth, labeled demonstrations, or a
gold-derived shortlist. Up to --max-attempts independent completions are drawn;
the first that passes deterministic validation (structure, leak phrases,
rationale length, and answer == ground truth under the SAME scorer RL uses) is
kept. Every attempted task writes one record with status accepted/exhausted/
error, so pass@1/pass@k and per-class yield come from the output file alone
(report_traces.py). Resume is by uid: accepted/exhausted skip, errors retry.

Open-response tasks (answer_style == "open": haptic_ts descriptions, tactile
notes/captions, free-text QA) have no exact-match gate and are skipped unless
--include-open. gen_sft_episodes imports the client helpers from here.

    python mirl_ext/sft/gen_sft_targets.py --tasks tasks.staged.jsonl \\
        --out traces.jsonl --image-root data/sft/media --dry-run
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import datetime
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
from mirl_ext.rewards import combined  # noqa: E402
from mirl_ext.rewards._common import extract_boxed_answer, format_reward  # noqa: E402
from mirl_ext.sft import teacher_context  # noqa: E402

BASE_URL = "http://point.dd.works:18890/v1"
# TRAPI needs the full dated deployment id (bare "gpt-5.6-sol" 404s).
DEFAULT_MODEL = "gpt-5.6-sol_2026-07-09"

MODE = "answer_blind_zero_shot"
MIN_THINK_CHARS, MAX_THINK_CHARS = 60, 3000
TRANSPORT_RETRIES = 3  # network retries per reasoning attempt

KEY_PATHS = [
    Path.home() / ".config/mirl/microsoft_openai_key",
    Path.home() / "mit/rlm-compaction/.env",
]

# Oracle-phrase guard (also imported by gen_sft_episodes).
_LEAK_RE = re.compile(
    r"\b(the (correct|given|provided|true|target) answer|we (are|were) told|"
    r"as (given|provided|stated above)|the label is|according to the (label|answer)|"
    r"since the answer)\b",
    re.IGNORECASE,
)

# Scaffolding/answer-conditioning phrases a self-contained zero-shot trace must
# never contain (the student sees only its own query).
_SCAFFOLD_LEAK_RE = re.compile(
    r"\b(the correct answer is|the provided answer|given the answer|ground truth|"
    r"i was told|support set|examples? above|the description says|the blue line)\b",
    re.IGNORECASE,
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
    """Explicit timeout, no SDK retries -- the caller owns retry policy."""
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=load_api_key(), timeout=timeout, max_retries=0)


def backoff(attempt: int) -> None:
    """Exponential + jitter. Callers skip it after the final attempt."""
    time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))


def read_done_uids(path: Path) -> set[str]:
    """All uids present in the output JSONL (tolerates a truncated tail)."""
    return set(read_status(path))


def read_status(path: Path) -> dict[str, str]:
    """uid -> last recorded status (records without one count as accepted)."""
    status: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    status[rec["uid"]] = rec.get("status", "accepted")
                except (json.JSONDecodeError, KeyError):
                    continue
    return status


def _resolve_image(raw, image_root):
    path = Path(raw)
    if image_root is not None:
        cand = Path(image_root) / path.parent.name / path.name
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
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ---- answer-blind zero-shot generation ----

@dataclasses.dataclass(frozen=True)
class TeacherTask:
    """Everything the request builder may see. Deliberately has NO ground-truth
    field; the answer key lives in a separate uid-keyed dict."""

    uid: str
    family: str
    data_source: str | None
    prompt: str
    image_paths: tuple[str, ...]
    frame_paths: tuple[str, ...]
    label_space: tuple[str, ...]
    staging_version: str | None

    @classmethod
    def from_row(cls, t: dict) -> "TeacherTask":
        return cls(
            uid=t["uid"],
            family=t["family"],
            data_source=t.get("data_source"),
            prompt=t["prompt"],
            image_paths=tuple(t.get("image_paths") or []),
            frame_paths=tuple(t.get("frame_paths") or []),
            label_space=tuple(t.get("label_space") or []),
            staging_version=t.get("staging_version"),
        )


class MediaMissing(Exception):
    pass


def build_request(task: TeacherTask, image_root, descriptions) -> tuple[list[dict], int]:
    """(chat messages, media count). Order: context, label set, question, media."""
    blocks = [teacher_context.family_context(task.family, descriptions)]
    if task.label_space:  # only exported when the prompt itself lists no options
        blocks.append(
            "Valid answers (copy one exactly):\n" + "\n".join(task.label_space)
        )
    question = re.sub(r"<(image|video|audio)>", "", task.prompt).strip()
    blocks.append(f"QUESTION:\n{question}")
    content: list[dict] = [{"type": "text", "text": "\n\n".join(blocks)}]

    media = task.frame_paths or task.image_paths  # frames staged oldest -> newest
    for raw in media:
        resolved = _resolve_image(raw, image_root)
        if resolved is None:
            raise MediaMissing(raw)
        content.append(_img(str(resolved)))
    return (
        [
            {"role": "system", "content": teacher_context.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        len(media),
    )


def validate(text: str, ground_truth: str, task: TeacherTask) -> tuple[bool, str, str | None]:
    """(accepted, reason, predicted). Deterministic; correctness is judged by the
    SAME per-family scorer RL uses (rewards.combined), so multilabel set
    semantics and label normalization match training exactly."""
    if not text:
        return False, "empty", None
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False, "format", None
    think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if not think:
        return False, "empty_think", None
    if len(think) < MIN_THINK_CHARS:
        return False, "think_too_short", None
    if len(think) > MAX_THINK_CHARS:
        return False, "think_too_long", None
    if _LEAK_RE.search(text) or _SCAFFOLD_LEAK_RE.search(text):
        return False, "leak", None
    if text.count("\\boxed{") != 1:
        return False, "no_boxed" if "\\boxed{" not in text else "multiple_boxed", None
    boxed = extract_boxed_answer(text)
    if boxed is None:
        return False, "no_boxed", None
    close = text.rfind("\\boxed{") + len("\\boxed{") + len(boxed)
    if text[close + 1 :].strip():
        return False, "text_after_boxed", None
    if format_reward(text) != 1.0:
        return False, "format", None
    predicted = _norm(boxed)
    if task.data_source == "ecg":
        # rewards.ecg matches category MENTIONS (lenient partial credit for RL) and
        # would accept "Abnormal" for "Normal"; the keep-gate needs the verbatim label.
        correct = predicted == _norm(ground_truth)
    else:
        correct = combined.compute_score(task.data_source, text, ground_truth)["acc"] == 1.0
    if not correct:
        space = {_norm(s) for s in task.label_space}
        reason = "wrong" if not space or predicted in space else "wrong_out_of_space"
        return False, reason, predicted
    return True, "ok", predicted


def generate_one(client, task: TeacherTask, ground_truth: str, args, descriptions) -> dict:
    """One record per task, whatever happens: accepted / exhausted / error."""
    record = {
        "uid": task.uid,
        "family": task.family,
        "row_index": int(task.uid.rsplit("#", 1)[1]) if "#" in task.uid else None,
        "data_source": task.data_source,
        "status": None,
        "model": args.model,
        "mode": MODE,
        "prompt_version": teacher_context.PROMPT_VERSION,
        "n_attempts": 0,
        "accepted_attempt": None,
        "response": None,
        "parsed_answer": None,
        "attempts": [],
        "error": None,
        "ground_truth": ground_truth,  # local audit only; never in the request
        "media_count": 0,
        "staging_version": task.staging_version,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        messages, n_media = build_request(task, args.image_root, descriptions)
    except MediaMissing as exc:
        record.update(status="error", error=f"media_missing:{exc}")
        return record
    record["media_count"] = n_media

    kwargs = {"model": args.model, "max_completion_tokens": args.max_completion_tokens}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    for attempt in range(1, args.max_attempts + 1):
        text = None
        for transport_try in range(TRANSPORT_RETRIES):  # transport != reasoning
            try:
                resp = client.chat.completions.create(messages=messages, **kwargs)
                text = (resp.choices[0].message.content or "").strip()
                break
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"api:{type(exc).__name__}"
                if transport_try + 1 < TRANSPORT_RETRIES:
                    backoff(transport_try)
        if text is None:
            record.update(status="error", n_attempts=attempt)
            return record
        record["error"] = None
        record["n_attempts"] = attempt
        good, reason, predicted = validate(text, ground_truth, task)
        if good:
            record.update(
                status="accepted", accepted_attempt=attempt, response=text, parsed_answer=predicted
            )
            return record
        record["attempts"].append({"attempt": attempt, "reason": reason, "predicted": predicted})
    record["status"] = "exhausted"
    return record


def dry_run(tasks: list[TeacherTask], args, descriptions) -> None:
    """Print one sanitized request per family. No API call, no ground truth."""
    seen: set[str] = set()
    for task in tasks:
        if task.family in seen:
            continue
        seen.add(task.family)
        try:
            messages, n_media = build_request(task, args.image_root, descriptions)
        except MediaMissing as exc:
            print(f"\n=== {task.family} ({task.uid}): MEDIA MISSING: {exc} ===")
            continue
        for part in messages[1]["content"]:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                part["image_url"]["url"] = f"<{len(url)} base64 chars redacted>"
        print(f"\n=== {task.family} ({task.uid}, media={n_media}) ===")
        print(json.dumps(messages, indent=2)[:6000])
        has_gt = any(f.name == "ground_truth" for f in dataclasses.fields(task))
        print(f"--- ground_truth field in request-visible task object: {has_gt}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument(
        "--include-open", action="store_true",
        help="also attempt answer_style=open tasks (no exact-match gate exists; "
        "they will almost never validate). Default: skip them.",
    )
    ap.add_argument(
        "--descriptions", type=Path, default=Path("data/sft/meta/text_description.json"),
        help="the 50 upstream SmellNet substance descriptions",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=4, help="independent reasoning attempts")
    ap.add_argument("--temperature", type=float, default=None, help="omit = endpoint default (1.0)")
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print one sanitized request per family; no API call")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    legacy = sum(1 for t in rows if "answer_style" not in t)
    if legacy:
        raise SystemExit(
            f"{legacy} tasks lack answer_style (legacy task file) -- re-export with "
            "export_sft_tasks.py first"
        )
    if args.families:
        rows = [t for t in rows if t["family"] in args.families]
    n_open = sum(1 for t in rows if t["answer_style"] == "open")
    if not args.include_open:
        rows = [t for t in rows if t["answer_style"] != "open"]
    answers = {t["uid"]: t["ground_truth"] for t in rows}
    tasks = [TeacherTask.from_row(t) for t in rows]

    descriptions = (
        teacher_context.load_descriptions(args.descriptions)
        if args.descriptions.is_file() and any(t.family == "smellnet_train" for t in tasks)
        else None
    )

    status = read_status(args.out)
    todo = [t for t in tasks if status.get(t.uid) in (None, "error")]
    random.Random(args.seed).shuffle(todo)  # --limit then samples across families
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"tasks={len(tasks)} skipped_open={n_open} done={sum(1 for s in status.values() if s != 'error')} "
        f"retry_errors={sum(1 for s in status.values() if s == 'error')} to_generate={len(todo)}"
    )
    if args.dry_run:
        dry_run(todo or tasks, args, descriptions)
        return
    if not todo:
        return

    # Fail before billing if any data_source has no reward scorer to validate with.
    bad = []
    for src in sorted({str(t.data_source) for t in todo}):
        try:
            combined.compute_score(src, "<think>x</think> \\boxed{y}", "y")
        except NotImplementedError:
            bad.append(src)
    if bad:
        raise SystemExit(f"no reward scorer for data_source(s) {bad} -- fix before generating")
    # A killed run can leave a partial last line; never fuse the next record onto it.
    if args.out.exists() and args.out.stat().st_size:
        with args.out.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            ends_clean = fh.read(1) == b"\n"
        if not ends_clean:
            with args.out.open("a") as fh:
                fh.write("\n")

    client = make_client(args.request_timeout)
    stats: collections.Counter = collections.Counter()
    t0 = time.time()
    # Append+flush per record (a kill must not lose paid work); as_completed so a
    # slow call never stalls the resume checkpoint.
    with args.out.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(generate_one, client, t, answers[t.uid], args, descriptions): t
            for t in todo
        }
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- never drop the other paid records
                t = futs[fut]
                rec = {"uid": t.uid, "family": t.family, "data_source": t.data_source,
                       "status": "error", "error": f"worker:{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats[(rec["family"], rec["status"])] += 1
            if done % 25 == 0:
                acc = sum(v for (_, s), v in stats.items() if s == "accepted")
                print(f"  {done}/{len(todo)} accepted={acc} {done / (time.time() - t0):.1f}/s")

    print(f"\n{'family':24s} {'accepted':>8s} {'exhausted':>9s} {'error':>6s}")
    for fam in sorted({f for f, _ in stats}):
        print(
            f"{fam:24s} {stats[(fam, 'accepted')]:8d} {stats[(fam, 'exhausted')]:9d} "
            f"{stats[(fam, 'error')]:6d}"
        )
    print(f"-> {args.out}  (run report_traces.py for pass@1/pass@k and per-class yield)")


if __name__ == "__main__":
    main()
