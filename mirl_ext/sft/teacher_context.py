"""Versioned answer-blind context the teacher sees before each query.

One fixed context string per family: what the recording physically is, which
observable properties matter, and (smellnet only) the 50 upstream substance
descriptions as weak class-level priors. Nothing here may depend on a query's
ground truth. Bump PROMPT_VERSION when any string changes; prompt_fingerprint()
is stored in every generation record so traces are attributable to the exact
prompt that produced them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_VERSION = "abz-v1"

SYSTEM_PROMPT = """\
You are a careful multimodal sensor analyst. Solve the classification or
question-answering task from the supplied query recording and fixed reference
definitions.

The reference definitions are uncertain domain priors, not observations or
labeled examples. Base the explanation on properties actually observable in
this query. Do not invent measurements, channel meanings, or events.

Give a concise evidence-based rationale. Identify the most important
low-level observations and briefly distinguish the strongest plausible
alternative when the evidence supports that comparison.

Do not mention a prompt, support set, examples, a provided answer, ground
truth, hidden labels, reference descriptions, plot colors, or image layout.

Return exactly:

<think>
A concise, query-grounded explanation.
</think>
\\boxed{CANONICAL_ANSWER}

Put nothing after the boxed answer. If the question lists options, the boxed
answer must copy one option verbatim (or the correct letters, comma-separated,
for select-all-that-apply questions)."""

# Only facts verified against the data/renderer: smellnet channel names are the
# CSV headers, ECG is 8 leads x 10 s @ 250 Hz drawn one panel per lead, tactile
# frames come from the visual-tactile glove videos, human_behaviour attaches
# frames plus the in-prompt transcript (its <audio> tag has no attached audio).
_FAMILY_CONTEXT = {
    "smellnet_train": (
        "The query is a multichannel MOX e-nose recording of one substance, shown "
        "as one panel per sensor channel (NO2, C2H5OH, VOC, CO, Alcohol, LPG) over "
        "time. Judge it by relative channel activation, onset and peak timing, "
        "plateau vs. decay shape, stability, and cross-channel relationships -- "
        "not absolute values."
    ),
    "ecg_train": (
        "The query is a resting ECG recording (8 leads, 10 seconds at 250 Hz), "
        "shown as one panel per lead. Judge rhythm regularity and rate, P/QRS/T "
        "presence and morphology, conduction intervals, ST-segment and T-wave "
        "changes, and which leads show them."
    ),
    "climb_train": (
        "The query is a medical imaging study; the question states the modality "
        "and the exact answer choices. When several images are attached they are "
        "views or slices of the same case, in order; video queries are frames "
        "sampled evenly in time from one scan or clip."
    ),
    "human_behaviour_train": (
        "The query is a human-interaction video, attached as frames sampled evenly "
        "in time (first to last). The spoken content appears as a transcript inside "
        "the question text; audio itself is not attached, and any <audio> tag in "
        "the question is formatting to ignore. Judge facial expression, gesture, "
        "posture, scene context, and the transcript's wording and tone."
    ),
    "tactile_train": (
        "The query is an interaction video from a sensorized-glove recording, "
        "attached as frames sampled evenly in time (first and last included). "
        "Frames may show the hand-object interaction and tactile pressure "
        "visualization. Judge contact timing, which fingers or palm engage, "
        "relative pressure and its stability, contact geometry, and object "
        "deformation. For select-all-that-apply questions answer with every "
        "correct option letter, comma-separated."
    ),
    "haptic_ts_train": (
        "The query is a tactile recording from a sensorized glove, shown as a "
        "taxel-activation heatmap over time plus a total-force trace. Judge when "
        "contact starts and ends, how pressure is distributed and evolves, and "
        "force stability."
    ),
}


def load_descriptions(path: str | Path) -> dict[str, str]:
    """The 50 upstream SmellNet substance descriptions (weak priors)."""
    desc = json.loads(Path(path).read_text())
    if len(desc) != 50:
        raise SystemExit(f"{path}: expected 50 substance descriptions, got {len(desc)}")
    return desc


def family_context(family: str, descriptions: dict[str, str] | None = None) -> str:
    ctx = _FAMILY_CONTEXT[family]
    if family == "smellnet_train" and descriptions:
        lines = [f"- {name}: {descriptions[name]}" for name in sorted(descriptions)]
        ctx += (
            "\n\nReference substance descriptions (uncertain class-level priors, "
            "not measured prototypes):\n" + "\n".join(lines)
        )
    return ctx


def prompt_fingerprint(descriptions: dict[str, str] | None = None) -> str:
    """Stable hash of everything that shapes the request text."""
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(SYSTEM_PROMPT.encode())
    for fam in sorted(_FAMILY_CONTEXT):
        h.update(family_context(fam, descriptions).encode())
    return h.hexdigest()[:16]
