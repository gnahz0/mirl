"""Diagnose why videos in data/skipped_hb.jsonl fail to decode.

Run on a compute node where /scratch/keane/ is mounted:
    python scripts/diagnose_skipped_videos.py
"""

import json
import logging
import os
import signal
import sys
import time
import warnings
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TORCHCODEC_LOG_LEVEL"] = "0"

SKIP_LIST = Path(__file__).resolve().parent.parent / "data" / "skipped_hb.jsonl"


class Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise Timeout()


def try_load(path: str, max_frames: int = 8, timeout_sec: int = 30) -> dict:
    """Try loading with torchcodec (via qwen_vl_utils), with SIGALRM timeout."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        from qwen_vl_utils import fetch_video
        sys.stderr = _stderr
        t0 = time.monotonic()
        video = fetch_video({"video": path, "max_frames": max_frames})
        total = time.monotonic() - t0
        if isinstance(video, tuple):
            video = video[0]
        return {"status": "ok", "shape": list(video.shape), "total_sec": round(total, 2)}
    except Timeout:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:120]}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    entries = []
    with open(SKIP_LIST) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if args.limit:
        entries = entries[:args.limit]

    print(f"Testing {len(entries)} skipped videos...\n")

    missing, ok, timeout, error = 0, 0, 0, 0

    for i, entry in enumerate(entries):
        vid = entry["video"]
        idx = entry.get("jsonl_line_index", "?")

        if not os.path.isfile(vid):
            missing += 1
            tag = "MISSING"
        else:
            size_kb = os.path.getsize(vid) / 1024
            dec = try_load(vid)
            s = dec["status"]
            if s == "ok":
                ok += 1
                tag = f"OK {dec['shape']} {dec['total_sec']}s {size_kb:.0f}KB"
            elif s == "timeout":
                timeout += 1
                tag = f"TIMEOUT {size_kb:.0f}KB"
            else:
                error += 1
                tag = f"ERROR {dec.get('error', '?')}"

        print(f"  [{i+1:>3}/{len(entries)}] {tag:40s} {os.path.basename(vid)}")
        sys.stdout.flush()

    print(f"\n=== Results ({len(entries)} total) ===")
    print(f"  OK:      {ok}")
    print(f"  MISSING: {missing}")
    print(f"  TIMEOUT: {timeout}")
    print(f"  ERROR:   {error}")
    


if __name__ == "__main__":
    main()
