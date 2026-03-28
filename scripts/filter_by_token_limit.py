"""Filter dataset entries by prompt token length.

Reads JSONL from data/, computes token length (including images/videos) with
Qwen3-VL processor, keeps only entries <= max_tokens, writes to new files.
Rows that fail decode / processor or exceed --row-timeout-sec are excluded.
On each problem, stderr prints ``idx=`` (0-based line index in the input JSONL) and resolved ``media:`` paths for a manual skip list (capture with e.g. ``2> filter_snags.log``).

Does not overwrite existing files; outputs to *_filtered_<N>.json

Usage:
  python scripts/filter_by_token_limit.py \\
      --check-source-datasets human_behaviour \\
      --max-tokens 8192 --max-video-frames 4

  python scripts/filter_by_token_limit.py --hb-only --max-tokens 8192 --max-video-frames 4

  # First-time download needs Hub once; after that workers use cache only (default) so respawns do not 429:
  HF_TOKEN=... python scripts/filter_by_token_limit.py ... --hf-online   # optional first run

  # Rerun: skip known-bad videos from data/skipped_hb.jsonl (default path); use --skip-hb-list \"\" to disable.

  # Periodic crash-safe snapshots (default every 500 rows): --checkpoint-every 1000 or 0 to disable.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Skip qwen_vl_utils's slow full-file torchvision fallback (skip row instead).
os.environ.setdefault("VERL_SKIP_QWENVL_VIDEO_TORCHVISION_FALLBACK", "1")

from PIL import Image
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.dataset.vision_utils import process_image, process_video


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _progress_line(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    w = 24
    filled = min(w, int(w * done / total + 0.5))
    _err(f"  [{'#' * filled}{'-' * (w - filled)}] {done:,}/{total:,} ({pct:.1f}%)")


def _atomic_write_jsonl(path: Path, rows: list) -> None:
    """Rewrite JSONL atomically so crash mid-write does not trash the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ent in rows:
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_kept_from_merged(merged: list, pass_through: list, hb_only: bool, max_tokens: int) -> list:
    """Build kept JSONL row list from sorted (idx, n, entry) rows + pass-through (non-hb_only only)."""
    if hb_only:
        index_to_entry: dict = {}
        for i, n, entry in merged:
            if n is not None and n <= max_tokens:
                index_to_entry[i] = entry
        return [index_to_entry[i] for i in sorted(index_to_entry.keys())]
    index_to_entry = {i: entry for i, entry in pass_through}
    for i, n, entry in merged:
        if n is not None and n <= max_tokens:
            index_to_entry[i] = entry
    return [index_to_entry[i] for i in sorted(index_to_entry.keys())]


def _resolve_media_path(raw: str, data_source_dir: str) -> str:
    p = str(raw)
    if p.startswith("file://"):
        p = p[7:]
    if p and not os.path.isabs(p):
        p = os.path.normpath(os.path.join(data_source_dir, p))
    return p


def entry_video_paths_norm(entry: dict, data_source_dir: str, video_key: str = "videos") -> list[str]:
    """Resolved, normpath'd video file paths for an entry (for skip-list matching)."""
    paths: list[str] = []
    for v in list(entry.get(video_key) or []):
        vd = dict(v) if isinstance(v, dict) else {"video": v}
        raw = vd.get("video")
        if isinstance(raw, list):
            for fp in raw:
                paths.append(os.path.normpath(_resolve_media_path(str(fp), data_source_dir)))
        elif raw is not None:
            paths.append(os.path.normpath(_resolve_media_path(str(raw), data_source_dir)))
    return paths


def load_skip_hb_video_paths(list_path: Path) -> set[str]:
    """Load data/skipped_hb.jsonl or .json (array); return set of normpath'd absolute video paths."""
    if not list_path.is_file():
        return set()
    out: set[str] = set()
    suf = list_path.suffix.lower()
    if suf == ".jsonl":
        with open(list_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                v = obj.get("video")
                if isinstance(v, str) and v:
                    out.add(os.path.normpath(v))
    elif suf == ".json":
        data = json.loads(list_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    v = obj.get("video")
                    if isinstance(v, str) and v:
                        out.add(os.path.normpath(v))
    return out


def entry_hit_skip_hb(entry: dict, data_source_dir: str, skip_paths: set[str]) -> bool:
    if not skip_paths:
        return False
    return any(p in skip_paths for p in entry_video_paths_norm(entry, data_source_dir))


def format_snag_media(entry: dict, data_source_dir: str, image_key: str = "images", video_key: str = "videos") -> str:
    """Resolved paths for stderr / manual skip lists (one logical path per semicolon)."""
    chunks: list[str] = []
    for v in list(entry.get(video_key) or []):
        vd = dict(v) if isinstance(v, dict) else {"video": v}
        raw = vd.get("video")
        if isinstance(raw, list):
            for k, fp in enumerate(raw):
                if k >= 12:
                    chunks.append(f"+{len(raw) - 12} more frames")
                    break
                chunks.append(_resolve_media_path(fp, data_source_dir))
        elif raw is not None:
            chunks.append(_resolve_media_path(str(raw), data_source_dir))
    for im in list(entry.get(image_key) or []):
        if isinstance(im, dict):
            raw = im.get("image")
            if isinstance(raw, str):
                chunks.append(_resolve_media_path(raw, data_source_dir))
    if not chunks:
        return "(no image/video paths on entry)"
    return "; ".join(chunks)


# ---------------------------------------------------------------------------
# Message building & token counting (unchanged core logic)
# ---------------------------------------------------------------------------

def build_messages(example: dict, prompt_key: str, image_key: str, video_key: str, data_source_dir: str):
    messages = list(example[prompt_key])
    images = list(example.get(image_key) or [])
    videos = list(example.get(video_key) or [])

    new_messages = []
    image_offset, video_offset = 0, 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            new_messages.append(dict(msg))
            continue

        content_list = []
        segments = [s for s in re.split("(<image>|<video>)", content) if s]
        for segment in segments:
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError(f"Missing image at offset {image_offset}")
                image = images[image_offset]
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    content_list.append({"type": "image", "image": image})
                elif isinstance(image, dict):
                    img = dict(image)
                    if "image" in img and not os.path.isabs(str(img.get("image", ""))):
                        img["image"] = os.path.join(data_source_dir, img["image"])
                    content_list.append({"type": "image", **img})
                else:
                    content_list.append({"type": "image", "image": image})
                image_offset += 1
            elif segment == "<video>":
                if video_offset >= len(videos):
                    raise ValueError(f"Missing video at offset {video_offset}")
                video = dict(videos[video_offset]) if isinstance(videos[video_offset], dict) else {"video": videos[video_offset]}
                if "video" in video and not os.path.isabs(str(video.get("video", ""))):
                    video["video"] = os.path.join(data_source_dir, video["video"])
                content_list.append({"type": "video", **video})
                video_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        new_messages.append({"role": msg["role"], "content": content_list})
    return new_messages


def compute_prompt_length(
    doc: dict,
    processor,
    prompt_key: str = "prompt",
    image_key: str = "images",
    video_key: str = "videos",
    data_source_dir: str = ".",
    image_patch_size: int = 14,
    max_video_frames: Optional[int] = None,
    snag_idx: Optional[int] = None,
) -> Optional[int]:
    """Returns token count or None on failure (row excluded)."""
    paths_source = doc
    try:
        doc = copy.deepcopy(doc)
        images_raw = list(doc.get(image_key) or [])
        videos_raw = list(doc.get(video_key) or [])

        messages = build_messages(doc, prompt_key, image_key, video_key, data_source_dir)
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        images = None
        if images_raw:
            images = [process_image(img, image_patch_size=image_patch_size) for img in images_raw]

        videos = None
        videos_kwargs = {}
        if videos_raw:
            capped = []
            for v in videos_raw:
                vc = dict(v) if isinstance(v, dict) else {"video": v}
                if max_video_frames is not None:
                    vc["max_frames"] = min(vc.get("max_frames", 768), max_video_frames)
                capped.append(vc)
            vs, metas = zip(
                *[process_video(v, image_patch_size=image_patch_size, return_video_metadata=True) for v in capped],
                strict=True,
            )
            videos = list(vs)
            videos_kwargs = {"video_metadata": list(metas), "do_sample_frames": False}

        inputs = processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)
        return len(inputs["input_ids"][0])
    except Exception as e:
        msg = str(e).replace("\n", " ")
        if len(msg) > 180:
            msg = msg[:177] + "..."
        media = format_snag_media(paths_source, data_source_dir, image_key, video_key)
        loc = f"idx={snag_idx} " if snag_idx is not None else ""
        _err(f"  Snag (decode {loc}{type(e).__name__}): {msg}\n    media: {media}")
        return None


def try_truncate_entry(
    entry: dict, processor, max_tokens: int, data_source_dir: str,
    max_video_frames: Optional[int],
    prompt_key: str = "prompt", image_key: str = "images", video_key: str = "videos",
    snag_idx: Optional[int] = None,
) -> Optional[tuple[int, dict]]:
    entry = copy.deepcopy(entry)
    messages = list(entry[prompt_key])

    def _length(e):
        return compute_prompt_length(
            e, processor, data_source_dir=data_source_dir,
            max_video_frames=max_video_frames, prompt_key=prompt_key,
            image_key=image_key, video_key=video_key, snag_idx=snag_idx,
        )

    n = _length(entry)
    if n is None:
        return None
    if n <= max_tokens:
        return (n, entry)

    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            if isinstance(msg.get("content"), str):
                messages[i] = {**msg, "content": "You are a helpful assistant."}
            elif isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        new_content.append({"type": "text", "text": "You are a helpful assistant."})
                    else:
                        new_content.append(item)
                messages[i] = {**msg, "content": new_content}
            entry = copy.deepcopy(entry)
            entry[prompt_key] = messages
            n = _length(entry)
            if n is None:
                return None
            if n <= max_tokens:
                return (n, entry)
            break

    messages_no_sys = [m for m in messages if m.get("role") != "system"]
    if messages_no_sys and len(messages_no_sys) < len(messages):
        entry = copy.deepcopy(entry)
        entry[prompt_key] = messages_no_sys
        n = _length(entry)
        if n is None:
            return None
        if n <= max_tokens:
            return (n, entry)

    while len(messages) > 2:
        if messages[0].get("role") == "system":
            messages = [messages[0]] + messages[3:]
        else:
            messages = messages[2:]
        if not messages:
            break
        entry = copy.deepcopy(entry)
        entry[prompt_key] = messages
        n = _length(entry)
        if n is None:
            return None
        if n <= max_tokens:
            return (n, entry)

    return None


# ---------------------------------------------------------------------------
# Worker: runs in a subprocess, loads processor once, processes rows from pipe.
# Parent can kill() this process if it hangs in native code.
# ---------------------------------------------------------------------------

def _worker_loop(model_name: str, conn, local_files_only: bool):
    # Default local_files_only=True: respawns after row-timeout must not call HF API (429 risk).
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_files_only
    )
    while True:
        try:
            task = conn.recv()
        except EOFError:
            break
        if task is None:
            break
        i, entry, data_dir, max_vf, max_tok, trunc = task
        try:
            n = compute_prompt_length(
                entry, processor, data_source_dir=data_dir, max_video_frames=max_vf, snag_idx=i
            )
            if n is not None and n > max_tok and trunc:
                result = try_truncate_entry(
                    entry, processor, max_tok, data_dir, max_vf, snag_idx=i
                )
                if result is not None:
                    n, entry = result
            conn.send((i, n, entry))
        except Exception as e:
            media = format_snag_media(entry, data_dir)
            msg = str(e).replace("\n", " ")[:200]
            _err(f"  Snag (worker crash idx={i} {type(e).__name__}): {msg}\n    media: {media}")
            conn.send((i, None, entry))


class _Worker:
    """Wraps a subprocess that processes one row at a time. Kill + respawn if stuck."""

    def __init__(self, model_name: str, local_files_only: bool):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._spawn()

    def _spawn(self):
        self.parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(
            target=_worker_loop,
            args=(self.model_name, child_conn, self.local_files_only),
            daemon=True,
        )
        self.proc.start()
        child_conn.close()
        self.task = None
        self.started_at = None

    @property
    def busy(self):
        return self.task is not None

    def elapsed(self) -> float:
        return (time.monotonic() - self.started_at) if self.started_at else 0.0

    def send(self, task: tuple):
        self.parent_conn.send(task)
        self.task = task
        self.started_at = time.monotonic()

    def poll(self) -> bool:
        return self.parent_conn.poll(0)

    def recv(self) -> tuple:
        result = self.parent_conn.recv()
        self.task = None
        self.started_at = None
        return result

    def kill_and_respawn(self) -> tuple:
        """Kill stuck worker, respawn, return the task it was working on."""
        task = self.task
        self.proc.kill()
        self.proc.join(timeout=5)
        self.parent_conn.close()
        self._spawn()
        return task

    def pop_dead_worker_task(self) -> Optional[tuple]:
        """If the child exited without sending (OOM, HF 429 on init, etc.), respawn and return its task."""
        if self.proc.is_alive() or self.task is None:
            return None
        if self.parent_conn.poll():
            return None
        task = self.task
        try:
            self.parent_conn.close()
        except Exception:
            pass
        self.proc.join(timeout=3)
        self._spawn()
        self.task = None
        self.started_at = None
        return task

    def shutdown(self):
        try:
            self.parent_conn.send(None)
            self.proc.join(timeout=10)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by prompt token length")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Input JSONL files (default: data/combined_{train,valid}_demo_only.json)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--only-check-hb", action="store_true",
                        help="Only token-check human_behaviour rows.")
    parser.add_argument("--check-source-datasets", nargs="+", default=None)
    parser.add_argument("--skip-source-datasets", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4). Each loads the VL processor.")
    parser.add_argument("--row-timeout-sec", type=float, default=10.0,
                        help="Kill + skip a row if it takes longer than this (default 10s). 0 disables.")
    parser.add_argument("--max-video-frames", type=int, default=None,
                        help="Hard cap on video frames per video (default 4).")
    parser.add_argument("--truncate-overlong", action="store_true")
    parser.add_argument("--subsample-pct", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hb-only", action="store_true",
                        help="Write only data/hb_only_filtered_<max_tokens>.json.")
    parser.add_argument("--write-hb-only", action="store_true")
    parser.add_argument("--hb-only-output", type=str, default=None)
    parser.add_argument(
        "--hf-online",
        action="store_true",
        help=(
            "Allow Hugging Face Hub API when loading the processor in each worker. "
            "Default is local cache only so worker respawns after timeouts do not hit the API (avoids 429)."
        ),
    )
    parser.add_argument(
        "--skip-hb-list",
        type=str,
        default=str(PROJECT_ROOT / "data" / "skipped_hb.jsonl"),
        help=(
            "JSONL or JSON array (e.g. data/skipped_hb.jsonl): rows whose resolved video path matches "
            "any ``video`` entry are excluded without token check. Use \"\" to disable."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help=(
            "Rewrite checkpoint JSONLs every N finished token-check rows for this input (0 disables). "
            "Files: <output_stem>_checkpoint<suffix> and HB: <hb_stem>_checkpoint_<input_stem><suffix>."
        ),
    )
    args = parser.parse_args()

    if args.max_video_frames is None:
        args.max_video_frames = int(os.environ.get("VIDEO_MAX_FRAMES", "4"))

    if args.inputs is None:
        data_dir = Path(PROJECT_ROOT) / "data"
        args.inputs = [
            str(data_dir / "combined_train_demo_only.json"),
            str(data_dir / "combined_valid_demo_only.json"),
        ]

    suffix = args.suffix or f"filtered_{args.max_tokens}"
    data_dir = args.data_dir or str(Path(args.inputs[0]).resolve().parent)
    row_to = float(args.row_timeout_sec)

    check_sources: frozenset[str]
    if args.hb_only or args.only_check_hb:
        check_sources = frozenset(["human_behaviour"])
    elif args.check_source_datasets:
        check_sources = frozenset(str(x) for x in args.check_source_datasets)
    else:
        check_sources = frozenset()

    skip_sources: set[str] = set()
    if args.skip_source_datasets:
        skip_sources.update(args.skip_source_datasets)

    def _resolve(p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else Path(PROJECT_ROOT) / path

    skip_combined_output = bool(args.hb_only)
    hb_only_path: Optional[Path] = None
    if args.hb_only:
        default_hb = Path(PROJECT_ROOT) / "data" / f"hb_only_filtered_{args.max_tokens}.json"
        hb_only_path = _resolve(args.hb_only_output) if args.hb_only_output else default_hb
    elif args.hb_only_output:
        hb_only_path = _resolve(args.hb_only_output)
    elif args.write_hb_only:
        hb_only_path = Path(PROJECT_ROOT) / "data" / f"hb_only_filtered_{args.max_tokens}.json"

    if hb_only_path is not None and "human_behaviour" not in check_sources:
        _err("Warning: HB-only output ignored (need --hb-only or --check-source-datasets human_behaviour).")
        hb_only_path = None

    hb_only_initialized = False

    def get_source_dataset(ent: dict) -> str:
        try:
            ei = ent.get("extra_info")
            if isinstance(ei, str):
                ei = json.loads(ei)
            return str((ei or {}).get("source_dataset", "") or "")
        except Exception:
            return ""

    def should_skip(ent) -> bool:
        if not skip_sources:
            return False
        return get_source_dataset(ent) in skip_sources

    skip_hb_paths: set[str] = set()
    if (args.skip_hb_list or "").strip():
        sp = Path(args.skip_hb_list.strip())
        if not sp.is_absolute():
            sp = PROJECT_ROOT / sp
        if not sp.is_file() and sp.suffix.lower() == ".jsonl":
            alt = sp.with_suffix(".json")
            if alt.is_file():
                sp = alt
        skip_hb_paths = load_skip_hb_video_paths(sp)
        if skip_hb_paths:
            _err(f"Loaded {len(skip_hb_paths):,} video paths from skip list {sp}")

    for input_path in args.inputs:
        input_path = Path(input_path)
        if not input_path.exists():
            _err(f"  Skip (not found): {input_path}")
            continue

        stem = input_path.stem
        out_path = input_path.parent / f"{stem}_{suffix}.json"

        entries = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if args.subsample_pct is not None:
            original_count = len(entries)
            rng = random.Random(args.seed)
            by_source = defaultdict(list)
            for i, e in enumerate(entries):
                by_source[e.get("data_source", "unknown")].append((i, e))
            subsampled = []
            for src, group in by_source.items():
                k = max(1, int(len(group) * args.subsample_pct / 100))
                subsampled.extend(rng.sample(group, min(k, len(group))))
            subsampled.sort(key=lambda x: x[0])
            entries = [e for _, e in subsampled]
            _err(f"  Subsampled {original_count:,} -> {len(entries):,} ({args.subsample_pct}%% per category)")

        pass_through = []
        to_compute = []
        preskipped_hb: list = []
        for i, entry in enumerate(entries):
            task = (i, entry, data_dir, args.max_video_frames, args.max_tokens, args.truncate_overlong)
            if check_sources:
                if get_source_dataset(entry) in check_sources:
                    if entry_hit_skip_hb(entry, data_dir, skip_hb_paths):
                        preskipped_hb.append((i, None, entry))
                    else:
                        to_compute.append(task)
                else:
                    pass_through.append((i, entry))
            elif should_skip(entry):
                pass_through.append((i, entry))
            else:
                if entry_hit_skip_hb(entry, data_dir, skip_hb_paths):
                    preskipped_hb.append((i, None, entry))
                else:
                    to_compute.append(task)

        total = len(to_compute)
        if check_sources:
            mode_info = f" (checking {sorted(check_sources)}; {len(pass_through):,} pass-through)"
        elif skip_sources:
            mode_info = f" (skipping {sorted(skip_sources)})"
        else:
            mode_info = ""
        _err(f"Processing {input_path}: {len(entries):,} entries, {total:,} to check{mode_info}")
        if preskipped_hb:
            _err(f"  Skip list: {len(preskipped_hb):,} rows excluded without decode (matched skipped_hb video path).")

        nw = max(1, args.workers)
        local_only = not args.hf_online
        if total > 0:
            _err(f"  {nw} worker(s), row timeout {row_to:g}s, max_video_frames {args.max_video_frames}")
            if local_only:
                _err("  Processor load: local HF cache only (--hf-online to allow Hub API).")

        prog_step = max(1, min(50, total // 250)) if total > 100 else 1
        results: list = []
        timed_out = 0
        worker_died = 0
        checkpoint_every = max(0, int(args.checkpoint_every))

        def save_checkpoint(tag_done: int, *, force: bool) -> None:
            if not checkpoint_every:
                return
            if not force and tag_done % checkpoint_every != 0:
                return
            mp = list(preskipped_hb) + list(results)
            if not mp:
                return
            mp.sort(key=lambda r: r[0])
            kept_p = build_kept_from_merged(mp, pass_through, args.hb_only, args.max_tokens)
            parts: list[str] = []
            if not skip_combined_output:
                ckpt = out_path.with_name(f"{out_path.stem}_checkpoint{out_path.suffix}")
                _atomic_write_jsonl(ckpt, kept_p)
                parts.append(f"{ckpt.name} ({len(kept_p):,} rows)")
            if hb_only_path is not None:
                hb_rows = (
                    kept_p
                    if args.hb_only
                    else [e for e in kept_p if get_source_dataset(e) == "human_behaviour"]
                )
                ckpt_hb = (
                    hb_only_path.parent
                    / f"{hb_only_path.stem}_checkpoint_{input_path.stem}{hb_only_path.suffix}"
                )
                _atomic_write_jsonl(ckpt_hb, hb_rows)
                parts.append(f"{ckpt_hb.name} ({len(hb_rows):,} HB)")
            if parts:
                _err(f"  checkpoint {tag_done:,}/{total:,} done: {'; '.join(parts)}")

        if total > 0:
            workers = [_Worker(args.model, local_files_only=local_only) for _ in range(nw)]
            task_iter = iter(to_compute)
            done = 0

            for w in workers:
                try:
                    w.send(next(task_iter))
                except StopIteration:
                    break

            while any(w.busy for w in workers):
                time.sleep(0.25)
                for w in workers:
                    if not w.busy:
                        continue
                    dead_task = w.pop_dead_worker_task()
                    if dead_task is not None:
                        worker_died += 1
                        i, ent, ddir = dead_task[0], dead_task[1], dead_task[2]
                        results.append((i, None, ent))
                        done += 1
                        media = format_snag_media(ent, ddir)
                        _err(
                            f"  Snag (worker died idx={i}, no result—OOM/HF error?): skipped row\n"
                            f"    media: {media}"
                        )
                        if done == 1 or done % prog_step == 0 or done == total:
                            _progress_line(done, total)
                        save_checkpoint(done, force=False)
                        try:
                            w.send(next(task_iter))
                        except StopIteration:
                            pass
                        continue
                    if w.poll():
                        results.append(w.recv())
                        done += 1
                        if done == 1 or done % prog_step == 0 or done == total:
                            _progress_line(done, total)
                        save_checkpoint(done, force=False)
                        try:
                            w.send(next(task_iter))
                        except StopIteration:
                            pass
                    elif row_to > 0 and w.elapsed() > row_to:
                        stuck_task = w.kill_and_respawn()
                        timed_out += 1
                        results.append((stuck_task[0], None, stuck_task[1]))
                        done += 1
                        idx, ent, ddir = stuck_task[0], stuck_task[1], stuck_task[2]
                        media = format_snag_media(ent, ddir)
                        _err(f"  Snag (timeout >{row_to:g}s, idx={idx}): killed worker, skipped row\n    media: {media}")
                        if done == 1 or done % prog_step == 0 or done == total:
                            _progress_line(done, total)
                        save_checkpoint(done, force=False)
                        try:
                            w.send(next(task_iter))
                        except StopIteration:
                            pass

            for w in workers:
                w.shutdown()

            if checkpoint_every:
                save_checkpoint(done, force=True)

            if timed_out:
                _err(f"  {timed_out:,} rows timed out and were skipped.")
            if worker_died:
                _err(f"  {worker_died:,} rows skipped after a worker process died (see Snag lines above).")

        results = preskipped_hb + results
        results.sort(key=lambda r: r[0])

        skipped_overlong = sum(1 for _, n, _ in results if n is not None and n > args.max_tokens)
        excluded = sum(1 for _, n, _ in results if n is None)
        kept = build_kept_from_merged(results, pass_through, args.hb_only, args.max_tokens)

        climb_str = ""
        if pass_through:
            if check_sources:
                climb_str = f", pass-through {len(pass_through):,}"
            elif skip_sources:
                climb_str = f", pass-through {len(pass_through):,}"

        if not skip_combined_output:
            with open(out_path, "w") as f:
                for entry in kept:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _err(f"  -> {out_path}: kept {len(kept):,}, overlong {skipped_overlong}, excluded {excluded}{climb_str}")
        else:
            _err(f"  -> (no combined file) kept {len(kept):,}, overlong {skipped_overlong}, excluded {excluded}{climb_str}")

        if hb_only_path is not None:
            hb_kept = kept if args.hb_only else [e for e in kept if get_source_dataset(e) == "human_behaviour"]
            mode = "a" if hb_only_initialized else "w"
            hb_only_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hb_only_path, mode) as hf:
                for entry in hb_kept:
                    hf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            hb_only_initialized = True
            _err(f"  -> {hb_only_path}: +{len(hb_kept):,} HB rows")


if __name__ == "__main__":
    main()
