"""Join accepted traces back to their split rows and emit veRL SFT parquet.

Per accepted uid: re-read the original SFT-half row, assert its ground truth
equals the one the trace was validated against (a shifted index would pair a
trace with the wrong media/label), keep the original prompt and every original
image/video in order, append exactly one assistant turn with the accepted
completion, and carry the original extra_info plus generation provenance.
Exhausted/error records and teacher-only staging never enter the parquet.

    python mirl_ext/sft/scripts/build_sft_parquet.py --traces data/sft/traces.jsonl --out .../sft_parquet
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mirl_ext.data.schema import (  # noqa: E402
    DATA_ROOT,
    config_path,
    extra_info,
    media_stem,
    prompt_messages,
)
from mirl_ext.sft.artifacts import (  # noqa: E402
    FrozenMediaIndex,
    frozen_media_index,
    source_row_fingerprint,
)
from mirl_ext.sft.traces import accepted_traces  # noqa: E402

# Generic-video fallback for untraced/open rows. Teacher-traced rows always use
# the exact frozen frame list instead (tactile 4..24; other RGB videos 8).
VIDEO_FRAMES = int(config_path("video_frames", "MIRL_VIDEO_FRAMES", "8"))


def sft_messages(row: dict) -> list[dict]:
    """Leading system turn merged into the user turn: MultiTurnSFTDataset
    tokenizes each message in isolation and Qwen3.5's template rejects a
    system-only list (tests/test_sft_pipeline.py::
    test_join_preserves_media_and_merges_system_turn covers this). KNOWN
    train/serve difference: GRPO renders a true system turn; here the same
    words ride at the head of the user turn."""
    msgs = prompt_messages(row)
    head = [m for m in msgs if m["role"] == "system"]
    if not head:
        return msgs
    rest = [m for m in msgs if m["role"] != "system"]
    merged = "\n\n".join([m["content"] for m in head] + [rest[0]["content"]])
    return [{"role": "user", "content": merged}] + rest[1:]


def build_record(
    row: dict,
    trace: dict,
    frozen: FrozenMediaIndex | None = None,
) -> dict:
    expected_source = trace.get("source_row_fingerprint")
    if not expected_source:
        raise ValueError(f"{trace['uid']}: trace has no source-row provenance")
    actual_source = source_row_fingerprint(
        row,
        str(trace["family"]),
        int(trace["row_index"]),
    )
    if actual_source != expected_source:
        raise ValueError(
            f"{trace['uid']}: source row changed since task export, refusing to write"
        )
    gt = (row.get("reward_model") or {}).get("ground_truth")
    if gt != trace["ground_truth"]:
        raise ValueError(f"{trace['uid']}: ground-truth join mismatch, refusing to write")
    # Minimal audit keys: join back to the trace file (which holds model,
    # prompt version, wrong guesses, ...) via uid when more detail is needed.
    # mode distinguishes verified (answer_blind_zero_shot) from coverage-tier
    # (answer_conditioned) traces.
    provenance = {
        "uid": trace["uid"],
        "ground_truth": gt,
        "accepted_attempt": trace.get("accepted_attempt"),
        "mode": trace.get("mode"),
        "source_row_fingerprint": expected_source,
        "staging_version": trace.get("staging_version"),
        "task_fingerprint": trace.get("task_fingerprint"),
    }
    # Set the config frame count and drop None-valued keys: qwen_vl_utils does
    # ele.get("min_frames", DEFAULT), and an explicit None defeats the default
    # and crashes frame sampling. With --media-from-staging, use the exact
    # frozen images/frames the teacher saw; qwen_vl_utils treats a path list as
    # a video, so training never seek-decodes the original mp4.
    images = []
    for image in row.get("images") or []:
        src = image.get("image") if isinstance(image, dict) else image
        if frozen is None:
            images.append(image)
            continue
        if not src:
            raise ValueError(f"{trace['uid']}: image entry has no source path")
        staged = frozen.images.get(media_stem(src))
        if not staged:
            raise ValueError(f"{trace['uid']}: no frozen staged image for {src!r}")
        images.append({**image, "image": staged} if isinstance(image, dict) else {"image": staged})

    videos = []
    for v in row.get("videos") or []:
        src = v.get("video") if isinstance(v, dict) else v
        if frozen is not None:
            if not src:
                raise ValueError(f"{trace['uid']}: video entry has no source path")
            staged = frozen.videos.get(media_stem(src))
            if not staged:
                raise ValueError(
                    f"{trace['uid']}: no frozen staged frames for video {src!r}"
                )
            videos.append({"video": staged})
            continue
        videos.append(
            {k: val for k, val in {**v, "max_frames": VIDEO_FRAMES}.items() if val is not None}
            if isinstance(v, dict) else v
        )
    return {
        "data_source": row.get("data_source"),
        "messages": sft_messages(row) + [{"role": "assistant", "content": trace["response"]}],
        "images": images,
        "videos": videos,
        "extra_info": json.dumps({**extra_info(row), **provenance}),
    }


def check_record(record: dict, uid: str) -> None:
    """Placeholder counts must match media counts or the SFT dataset asserts."""
    text = "".join(m["content"] for m in record["messages"])
    n_img, n_vid = text.count("<image>"), text.count("<video>")
    assert n_img == len(record["images"]), f"{uid}: {n_img} <image> vs {len(record['images'])}"
    assert n_vid == len(record["videos"]), f"{uid}: {n_vid} <video> vs {len(record['videos'])}"
    assert record["messages"][-1]["role"] == "assistant" and record["messages"][-1]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traces", nargs="*", default=[])
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    ap.add_argument("--out", default=f"{DATA_ROOT}/split_grpo/sft_parquet")
    ap.add_argument(
        "--open-gt",
        action="store_true",
        help="also emit <family>_open_sft.parquet where open-response rows train "
        "directly on their ground-truth text (captions/answers; no teacher)",
    )
    ap.add_argument(
        "--media-from-staging",
        "--frames-from-staging",
        dest="media_from_staging",
        type=Path,
        default=None,
        help="frozen-media root; rows train on the exact images/video frames "
        "seen by the teacher; missing media are fatal",
    )
    args = ap.parse_args()
    if not args.traces and not args.open_gt:
        ap.error("nothing to build: pass --traces and/or --open-gt")
    if args.traces and args.media_from_staging is None:
        ap.error("--media-from-staging is required when building teacher-traced rows")

    import pyarrow as pa
    import pyarrow.parquet as pq

    traces = accepted_traces([Path(p) for p in args.traces])
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for t in traces.values():
        by_family[t["family"]].append(t)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    total = 0

    for family, items in sorted(by_family.items()):
        rows = pq.read_table(Path(args.split_root) / "sft" / f"{family}.parquet").to_pylist()
        records = []
        frozen = (
            frozen_media_index(args.media_from_staging / family)
            if args.media_from_staging
            else None
        )
        for t in items:
            row = rows[t["row_index"]]
            record = build_record(row, t, frozen)
            check_record(record, t["uid"])
            records.append(record)
        total += len(records)
        n_img = sum(1 for r in records if r["images"])
        n_vid = sum(1 for r in records if r["videos"])
        print(
            f"{family:22s} accepted={len(items):6d} written={len(records):6d} "
            f"with_images={n_img} with_videos={n_vid}"
        )
        pq.write_table(pa.Table.from_pylist(records), out_root / f"{family}_sft.parquet")
        # Older builds held out 2% for trainer val/loss. All accepted SFT-half
        # rows now train; remove the obsolete derived shard after replacing the
        # train parquet so a launcher or audit cannot accidentally reuse it.
        stale_val = out_root / f"{family}_sft_val.parquet"
        if stale_val.exists():
            stale_val.unlink()

    print(f"\nwrote {total} rows across {len(by_family)} files -> {out_root}")

    if args.open_gt:
        from mirl_ext.data.schema import FAMILIES, OPEN_SOURCES

        for family in FAMILIES:
            src = Path(args.split_root) / "sft" / f"{family}.parquet"
            if not src.exists():
                continue
            open_frozen = (
                frozen_media_index(args.media_from_staging / family)
                if args.media_from_staging
                else None
            )
            records = []
            for i, row in enumerate(pq.read_table(src).to_pylist()):
                gt = str((row.get("reward_model") or {}).get("ground_truth") or "")
                if str(row.get("data_source")) not in OPEN_SOURCES or not gt.strip():
                    continue
                # The ground-truth text IS the target; reuse the trace path with
                # the caption standing in as the "response".
                record = build_record(
                    row,
                    {
                        "uid": f"{family}#{i}",
                        "family": family,
                        "row_index": i,
                        "source_row_fingerprint": source_row_fingerprint(row, family, i),
                        "ground_truth": gt,
                        "response": gt,
                    },
                    open_frozen,
                )
                check_record(record, f"{family}#{i}")
                records.append(record)
            if records:
                pq.write_table(
                    pa.Table.from_pylist(records), out_root / f"{family}_open_sft.parquet"
                )
                print(
                    f"{family:22s} open-gt rows={len(records):6d} -> {family}_open_sft.parquet"
                )
            stale_val = out_root / f"{family}_open_sft_val.parquet"
            if stale_val.exists():
                stale_val.unlink()

    print("Smoke-test with a tiny trainer run (data.train_max_samples=16) before training.")


if __name__ == "__main__":
    main()
