"""Single home for MIRL row/domain facts shared across alignment, SFT, and RL.

Everything here is a constant or a tiny stdlib-only helper: how a row is shaped
(prompt is a MESSAGE LIST, extra_info is a JSON STRING, media entries are
dicts), which data sources belong to which task family, and the recording-stem
join key. Stages keep their own datasets and pipelines; only the facts live
here, so they can never drift apart again.
"""

from __future__ import annotations

import hashlib
import json
import os
import string
from dataclasses import dataclass
from pathlib import Path

_CONFIG = Path(__file__).parents[1] / "sft" / "config.json"


def config_path(key: str, env: str, fallback: str) -> str:
    """Resolve env (mirl.env) -> sft/config.json -> fallback; cluster paths never live in code."""
    if os.environ.get(env):
        return os.environ[env].rstrip("/")
    if _CONFIG.is_file():
        cfg = json.loads(_CONFIG.read_text())
        if key in cfg:
            return str(cfg[key]).rstrip("/")
    return fallback


DATA_ROOT = config_path("cluster_data_root", "MIRL_DATA_ROOT", "data")
SCRATCH_ROOT = config_path("cluster_scratch_root", "MIRL_SCRATCH_ROOT", "scratch")


def media_stem(path) -> str:
    """FROZEN staged-media stem: sha1 of the source-path string, 20 hex chars.
    On-disk staged filenames and --frames-from-staging joins depend on it —
    never change this hash."""
    return hashlib.sha1(str(path).encode()).hexdigest()[:20]


def iter_jsonl(path):
    """Strict JSONL reader: parse every non-blank line, fail loud on a malformed
    one (task files are fully machine-written). Trace files instead need the
    lenient last-record-per-uid readers in sft/scripts."""
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

FAMILIES = [
    "ecg_train",
    "haptic_ts_train",
    "climb_train",
    "human_behaviour_train",
    "tactile_train",
]

# Free-text sources (checked once against the data): captions/notes and open QA
# whose answers can't be exact-match graded. Everything else is closed.
OPEN_SOURCES = {
    "haptic_tactile",                                         # haptic_ts descriptions
    "description", "tactile_description", "mat_description",  # tactile captions/notes
    "part_notes", "objA_notes", "objB_notes", "deformation_note",
    "intentqa", "siq2", "mimeqa",                             # free-text video QA
}

@dataclass(frozen=True)
class TactileTaskSpec:
    """Closed tactile task semantics shared by parsing, training, and metrics."""

    name: str
    labels: tuple[str, ...]
    min_choices: int = 1
    max_choices: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.labels:
            raise ValueError("a tactile task needs a name and at least one label")
        if not 1 <= self.min_choices <= self.max_choices <= len(self.labels):
            raise ValueError(
                f"invalid choice cardinality for {self.name}: "
                f"{self.min_choices}..{self.max_choices} over {len(self.labels)} labels"
            )

    @property
    def multilabel(self) -> bool:
        return self.max_choices > 1


# The six closed tactile QA tasks are declared once here. Every compatibility
# view below (label banks, spans, multilabel membership) is derived from these
# specs so Stage-1, SFT, reward routing, and metrics cannot disagree silently.
TACTILE_TASK_SPECS: tuple[TactileTaskSpec, ...] = (
    TactileTaskSpec(
        "initial_fingers",
        (
            "initial contact: thumb",
            "initial contact: index finger",
            "initial contact: middle finger",
            "initial contact: ring finger",
            "initial contact: pinky finger",
            "initial contact: palm",
        ),
        max_choices=6,
    ),
    TactileTaskSpec(
        "highest_pressure",
        (
            "highest pressure: thumb",
            "highest pressure: index finger",
            "highest pressure: middle finger",
            "highest pressure: ring finger",
            "highest pressure: pinky finger",
            "highest pressure: palm",
        ),
        max_choices=6,
    ),
    TactileTaskSpec(
        "force_level",
        (
            "force level: light, under 5 newtons",
            "force level: moderate, 5 to 10 newtons",
            "force level: firm, 10 to 20 newtons",
            "force level: strong, over 20 newtons",
        ),
    ),
    TactileTaskSpec(
        "grip_stability",
        (
            "grip stability: stable",
            "grip stability: unstable",
        ),
    ),
    TactileTaskSpec(
        "contact_feature",
        (
            "contact geometry: edge",
            "contact geometry: flat surface",
            "contact geometry: curved surface",
            "contact geometry: corner",
            "contact geometry: multiple edges",
            "contact geometry: edge and surface",
            "contact geometry: transitioning from edge to surface",
            "contact geometry: complex geometry with multiple features",
        ),
    ),
    TactileTaskSpec(
        "local_shape",
        (
            "local surface shape: flat",
            "local surface shape: convex",
            "local surface shape: concave",
            "local surface shape: edge",
        ),
    ),
)
TACTILE_TASKS: dict[str, TactileTaskSpec] = {spec.name: spec for spec in TACTILE_TASK_SPECS}
if len(TACTILE_TASKS) != len(TACTILE_TASK_SPECS):
    raise ValueError("duplicate tactile task name")

# Compatibility names retained for call sites that only need one projection.
TACTILE_TASK_LABELS: dict[str, tuple[str, ...]] = {name: spec.labels for name, spec in TACTILE_TASKS.items()}
MULTILABEL_TASKS = frozenset(name for name, spec in TACTILE_TASKS.items() if spec.multilabel)

TACTILE_TASK_SPANS: dict[str, tuple[int, int]] = {}
_tactile_offset = 0
for _tactile_spec in TACTILE_TASK_SPECS:
    TACTILE_TASK_SPANS[_tactile_spec.name] = (
        _tactile_offset,
        _tactile_offset + len(_tactile_spec.labels),
    )
    _tactile_offset += len(_tactile_spec.labels)
TACTILE_NUM_LABELS = _tactile_offset
del _tactile_offset, _tactile_spec


def parse_tactile_answer(answer: object, task: str) -> tuple[int, ...]:
    """Parse one letter answer into sorted label IDs and enforce task cardinality.

    All tasks use the same set-valued representation: a single-label answer is
    just a set with one member. Task specs decide only how many unique choices
    are legal; they do not fork parsing or target construction.
    """
    try:
        spec = TACTILE_TASKS[task]
    except KeyError as exc:
        raise ValueError(f"unknown tactile task {task!r}") from exc

    choices = [choice.strip() for choice in str(answer).split(",")]
    if any(len(choice) != 1 or choice not in string.ascii_uppercase for choice in choices):
        raise ValueError(f"invalid answer {answer!r} for {task}")
    indices = tuple(sorted({string.ascii_uppercase.index(choice) for choice in choices}))
    if any(index >= len(spec.labels) for index in indices):
        raise ValueError(f"invalid answer {answer!r} for {task}")
    if not spec.min_choices <= len(indices) <= spec.max_choices:
        expected = (
            str(spec.min_choices)
            if spec.min_choices == spec.max_choices
            else f"{spec.min_choices}..{spec.max_choices}"
        )
        raise ValueError(
            f"{task} expects {expected} unique choice(s), got {len(indices)} in {answer!r}"
        )
    return indices

# Reward-routing source sets (rewards/combined.py dispatches on these).
TACTILE_SOURCES = {
    "verify", "initial_fingers", "highest_pressure", "more_deformable",
    "deformation_type", "deformation_note", "objA_texture", "objB_texture",
    "objA_notes", "objB_notes", "grasp_location", "contact_feature",
    "local_shape", "grip_stability", "future_stability", "force_level",
    "shear_direction", "object_motion", "fail_reason", "fail_improvement",
    "description", "mat_description", "part_notes", "tactile_description",
}
HUMAN_BEHAVIOUR_SOURCES = {
    "cremad", "chsimsv2", "daicwoz", "intentqa", "meld_emotion", "meld_senti",
    "mimeqa", "mmpsy_anxiety", "mmpsy_depression", "mmsd", "mosei_emotion",
    "mosei_senti", "ptsd_in_the_wild", "siq2", "tess", "urfunny",
}
MEDICAL_SOURCES = {"chest_xray", "ct", "derm", "fundus", "mammo", "mri", "pathology", "ultrasound"}

# The taxonomies above describe one reality; drift between them was a real bug class.
assert set(TACTILE_TASK_LABELS) <= TACTILE_SOURCES
assert not (OPEN_SOURCES & set(TACTILE_TASK_LABELS))


def prompt_messages(row: dict) -> list[dict]:
    """The row's FULL prompt message list -- NOT prompt[0]: climb/tactile
    carry system + user turns, and dropping the user turn loses the question and
    the <image>/<video> placeholder (this bug shipped once)."""
    return [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]


def prompt_text(row: dict) -> str:
    """All message contents joined -- what a text-only API call should receive."""
    return "\n\n".join(m["content"] for m in prompt_messages(row) if m["content"])


def extra_info(row: dict) -> dict:
    """extra_info is stored as a JSON STRING in every MIRL parquet."""
    ei = row.get("extra_info")
    if isinstance(ei, str):
        ei = json.loads(ei)
    return ei if isinstance(ei, dict) else {}


def media_refs(row: dict) -> tuple[list[str], str]:
    """(image paths in original order, video path)."""
    images = []
    for entry in row.get("images") or []:
        images.append(entry.get("image", "") if isinstance(entry, dict) else str(entry))
    video_path = ""
    videos = row.get("videos") or []
    if videos:
        first = videos[0]
        video_path = (first.get("video") or "") if isinstance(first, dict) else str(first)
    return [p for p in images if p], video_path


def first_media_path(row: dict) -> str:
    """First media reference of any kind (the split's group key for path-mode)."""
    for key in ("signals", "images", "videos"):
        entries = row.get(key)
        if entries:
            entry = entries[0]
            if isinstance(entry, dict):
                for field in ("signal", "image", "video", "path"):
                    if entry.get(field):
                        return str(entry[field])
                return json.dumps(entry, sort_keys=True)
            return str(entry)
    return ""


def recording_stem(row: dict) -> str | None:
    """The shared 3DHaptic recording id: extra_info.stem, else the video-path
    stem. One recording appears as tactile-video rows AND a haptic time-series
    row; this key is what keeps them on the same side of every split."""
    ei = extra_info(row)
    stem = ei.get("stem")
    if stem:
        return str(stem)
    video_path = ei.get("video_path")
    if video_path:
        return Path(str(video_path)).stem
    return None
