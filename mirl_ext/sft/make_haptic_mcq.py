"""Mint gradable MCQ rows for the tactile TIME-SERIES view.

Upstream attached the six closed tactile questions only to the VIDEO rows; the
time-series (haptic_ts) rows got just the open caption, so that view has no
gradable supervision. The answers are per-recording, so this joins the
SFT-half tactile MCQ rows to the SFT-half haptic_ts rows by recording stem and
emits standard-schema rows: same question text and answer, but the media is
the tactile time-series plot. Output split_grpo/sft/haptic_mcq_train.parquet
then flows through the normal export -> stage -> generate -> build pipeline.
Stem-locking in the split keeps this leak-free (a recording's video and ts
rows are always on the same side).

    srun -p cpu -c 4 --mem=32G <env>/bin/python mirl_ext/sft/make_haptic_mcq.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.schema import (  # noqa: E402
    DATA_ROOT,
    TACTILE_MCQ_SOURCES as MCQ_SOURCES,
    prompt_messages,
    recording_stem,
)

# Mirrors the tactile system prompt's structure verbatim, with the modality
# sentence swapped from video to the pressure plot.
SYSTEM = (
    "You are an expert in analyzing tactile pressure recordings from a sensorized "
    "glove. The provided image shows taxel activation over time and the total "
    "force curve. Examine the pressure patterns carefully and answer the "
    "multiple-choice question. You FIRST think about the reasoning process as an "
    "internal monologue and then provide the final answer. The reasoning process "
    "MUST BE enclosed within <think> </think> tags. The final answer MUST BE a "
    "single letter (or comma-separated letters for multi-select) wrapped in "
    "\\boxed{}."
)


def mint_row(tactile_row: dict, plot_path: str) -> dict:
    """The tactile MCQ re-asked over the time-series plot."""
    user = next(m for m in prompt_messages(tactile_row) if m["role"] == "user")
    question = user["content"].replace("<video>", "").strip()
    gt = (tactile_row.get("reward_model") or {}).get("ground_truth")
    return {
        "data_source": tactile_row.get("data_source"),
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<image>\n{question}"},
        ],
        "images": [{"image": plot_path}],
        "videos": [],
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": json.dumps({
            "stem": recording_stem(tactile_row),
            "question_type": tactile_row.get("data_source"),
            "minted_from": "tactile_train answers x haptic_ts_train plot",
        }),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-root", default=f"{DATA_ROOT}/split_grpo")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    sft = Path(args.split_root) / "sft"
    plots: dict[str, str] = {}
    for row in pq.read_table(sft / "haptic_ts_train.parquet").to_pylist():
        images = row.get("images") or []
        stem = recording_stem(row)
        if images and stem:
            plots[stem] = images[0]["image"]

    minted, seen = [], set()
    for row in pq.read_table(sft / "tactile_train.parquet").to_pylist():
        src, stem = str(row.get("data_source")), recording_stem(row)
        if src not in MCQ_SOURCES or stem not in plots or (stem, src) in seen:
            continue
        seen.add((stem, src))
        minted.append(mint_row(row, plots[stem]))

    out = sft / "haptic_mcq_train.parquet"
    pq.write_table(pa.Table.from_pylist(minted), out)
    per_src = {s: sum(1 for m in minted if m["data_source"] == s) for s in MCQ_SOURCES}
    print(f"minted {len(minted)} rows over {len(plots)} recordings -> {out}\n{per_src}")


if __name__ == "__main__":
    main()
