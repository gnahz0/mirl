# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Build labelled reference galleries (contact sheets) of time-series plots.

Why sheets and not separate images: the endpoint enforces a HARD cap of **50 images
per request** (measured: 51 images -> 400 "Too many images in request"). A 50-class
SmellNet gallery plus the query is 51, so one-image-per-class is not merely expensive,
it is impossible. The cap counts IMAGES, not pixels, so tiling many plots into one
sheet buys both headroom and tokens. Measured per-plot prompt-token cost:

    separate full-size plot            578
    2x2 sheet  (4 plots, 1100px wide)  220
    3x3 sheet  (9 plots, 1299px wide)   94
    4x4 sheet (16 plots, 1500px wide)   53

Cost is flat (~847 tokens) for any sheet above ~1300px, so the real constraint is
panel legibility, not tokens. 3x3 keeps each SmellNet panel at 433x518, which
preserves the cross-channel rise/plateau shapes the task actually depends on.

Two distinct uses, and they are NOT the same job:

* **Blind eval** -- the gallery is the only information about what each class looks
  like. Tests whether the plots carry class-discriminative signal at all.
* **Trace generation** -- the answer is already given, so the gallery does not help
  GPT "get it right". What it does is let the reasoning be COMPARATIVE ("the NO2
  plateau sits above the clove reference and rises more slowly, matching almond")
  rather than a bare description followed by an unfounded assertion. That is a
  strictly better SFT target: it teaches a comparison procedure instead of a claim.

Gallery rows are returned so the caller can EXCLUDE them from its query set --
scoring a gallery example as its own query is trivially correct and would inflate
every number.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_image(image_path: str, image_root: Path | None) -> Path | None:
    """Map a cluster-absolute plot path onto a local staging dir if one is given."""
    p = Path(image_path)
    if image_root is not None:
        cand = image_root / p.parent.name / p.name
        if cand.is_file():
            return cand
    return p if p.is_file() else None


def build_sheet(
    items: list[tuple[Path, str]],
    out_path: Path,
    grid: int = 3,
    panel_w: int = 433,
    caption_h: int = 30,
) -> Path:
    """Tile ``items`` [(image, caption)] into one captioned contact sheet.

    Captions are drawn INSIDE the sheet rather than sent as adjacent text, so a
    panel can never drift away from its label -- the alternative (image, text, image,
    text, ...) relies on the model tracking interleaved order across dozens of parts.
    """
    from PIL import Image, ImageDraw

    if not items:
        raise ValueError("no items for sheet")
    with Image.open(items[0][0]) as probe:
        aspect = probe.height / probe.width
    panel_h = int(panel_w * aspect)
    cell_h = panel_h + caption_h
    rows = (len(items) + grid - 1) // grid

    sheet = Image.new("RGB", (panel_w * grid, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(13, caption_h - 12))

    for i, (path, caption) in enumerate(items):
        cx, cy = (i % grid) * panel_w, (i // grid) * cell_h
        with Image.open(path) as im:
            sheet.paste(im.convert("RGB").resize((panel_w, panel_h)), (cx, cy))
        draw.rectangle([cx, cy + panel_h, cx + panel_w - 1, cy + cell_h - 1], fill="#111111")
        text = caption if len(caption) <= 46 else caption[:43] + "..."
        draw.text((cx + 6, cy + panel_h + 5), text, fill="white", font=font)
        draw.rectangle([cx, cy, cx + panel_w - 1, cy + cell_h - 1], outline="#888888")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Per-family reference selection
# ---------------------------------------------------------------------------

def _label_of(task: dict) -> str:
    return str(task.get("ground_truth", ""))


def _is_mixture(task: dict) -> bool:
    return any(c.isdigit() for c in _label_of(task))


def select_references(
    tasks: list[dict],
    family: str,
    per_class: int,
    seed: int = 0,
    allowed_sources: set[str] | None = None,
) -> tuple[list[dict], str]:
    """Choose gallery rows for one family, and describe what the gallery means.

    The selection differs per family because "class" does:

    * ``ecg_train``       -- 7 diagnostic categories; plenty of examples each.
    * ``smellnet_train``  -- base rows are 50-way substance ID. Mixture rows are a
      DIFFERENT task on a DIFFERENT rig (4 channels @ 10 Hz vs 6 @ 1 Hz) with
      physically different materials (extracts, not foods), so base plots are NOT
      valid references for a mixture query. For mixtures the right references are the
      12 PURE odorants recorded on the mixture rig, which do exist in our data
      (147 rows, all 12 of the paper's palette).
    * ``haptic_ts_train`` -- free-text description, one unique label per row, so
      there is no class to demonstrate. The gallery teaches DESCRIPTION STYLE instead.
    """
    rng = random.Random(seed)
    pool = [t for t in tasks if t["family"] == family]
    if allowed_sources:
        pool = [t for t in pool if t.get("data_source") in allowed_sources]

    if family == "smellnet_train":
        base = [t for t in pool if t.get("data_source") == "smellnet_base"]
        pure = [t for t in pool if t.get("data_source") == "smellnet_mixture" and not _is_mixture(t)]
        by_label: dict[str, list[dict]] = collections.defaultdict(list)
        for t in base:
            by_label[_label_of(t)].append(t)
        for t in pure:
            by_label[f"[pure] {_label_of(t)}"].append(t)
        note = (
            "Each panel is one labelled recording. Panels labelled '[pure] X' are the "
            "single odorant X recorded on the 4-channel mixture rig; the others are "
            "6-channel base-substance recordings."
        )
    elif family == "haptic_ts_train":
        by_label = {f"example {i+1}": [t] for i, t in enumerate(rng.sample(pool, min(len(pool), 8)))}
        note = "Each panel is one recording with the description it should receive."
    else:
        by_label = collections.defaultdict(list)
        for t in pool:
            by_label[_label_of(t)].append(t)
        note = "Each panel is one labelled recording of the stated class."

    # Candidates carry "_pri" (0 = first --gallery-tasks file, 1 = next, ...). Sorting
    # by it exhausts the preferred pool before borrowing: references should come from
    # the RL half so SFT rows stay queryable, and only classes the RL half cannot cover
    # (21 of 50 base substances have just 2 rows there) spend an SFT row.
    picked: list[dict] = []
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda t: (t.get("_pri", 0), rng.random()))
        picked.extend(items[:per_class])
    return picked, note


def build_gallery(
    tasks: list[dict],
    family: str,
    per_class: int,
    cache_dir: Path,
    image_root: Path | None,
    grid: int = 3,
    panel_w: int = 433,
    caption_key: str = "ground_truth",
    allowed_sources: set[str] | None = None,
) -> tuple[list[Path], list[dict], str]:
    """Build (or load from cache) the sheets for one family.

    Returns (sheet paths, the reference RECORDS, human note). Records, not uid strings:
    uid is "<family>#<row_index>" and row_index restarts per split half, so a set of
    uids silently merges distinct rows from different pools -- it collapsed a 150-row
    3-per-class selection to 129 and would under-exclude query rows. Cache key covers every
    input that changes the pixels, so a changed per_class or grid cannot silently
    serve a stale sheet.
    """
    refs, note = select_references(
        tasks, family, per_class, allowed_sources=allowed_sources
    )
    if not refs:
        return [], set(), note

    key = hashlib.sha1(
        json.dumps(
            [family, per_class, grid, panel_w, [r["uid"] for r in refs]], sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    out_dir = cache_dir / f"{family}_pc{per_class}_g{grid}_{key}"

    items: list[tuple[Path, str]] = []
    skipped: list[str] = []
    for r in refs:
        path = resolve_image(r.get("image_path", ""), image_root)
        if path is None:
            # Silently dropping an unresolvable reference makes the sheet disagree with
            # what the caller believes is on it -- which is exactly how a smoke-test
            # caption check "fails" while the sheet is actually fine. Record and report.
            skipped.append(r.get("uid", "?"))
            continue
        cap = r.get(caption_key) or ""
        if r["family"] == "smellnet_train" and r.get("data_source") == "smellnet_mixture":
            cap = f"[pure] {cap}"
        items.append((path, str(cap)))

    if skipped:
        print(
            f"  [gallery:{family}] WARNING: {len(skipped)} reference plot(s) not found "
            f"locally and omitted from the sheets (e.g. {skipped[:3]}). "
            f"Stage them or they are simply absent as references."
        )

    per_sheet = grid * grid
    sheets: list[Path] = []
    for i in range(0, len(items), per_sheet):
        out = out_dir / f"sheet_{i // per_sheet:02d}.png"
        if not out.is_file():
            build_sheet(items[i : i + per_sheet], out, grid=grid, panel_w=panel_w)
        sheets.append(out)

    # Return the RESOLVED captions in sheet order, so a verifier checks what was
    # actually rendered rather than what was requested.
    build_gallery.last_captions = [cap for _p, cap in items]
    return sheets, refs, note


# Placement note: images ARE accepted in a system message on this endpoint (tested).
# The gallery still goes in a leading USER turn, because that is the arrangement the
# smoke test verifies end to end -- asked to read back the top row, the user-turn form
# returns the captions ("allspice, allspice, allspice, almond") while the system-turn
# form drifted into describing the panels instead. Same information, different attention.
GALLERY_INTRO = (
    "REFERENCE GALLERY. The following sheet(s) show labelled example recordings, "
    "rendered exactly like the one you will be asked about. Each panel's label is "
    "printed in the dark band directly beneath it. {note}\n"
    "Study the panels: what distinguishes classes is the JOINT pattern across "
    "channels/leads -- relative rise rates, plateau levels, timing and morphology -- "
    "not any single channel's absolute value, since baselines drift between sessions."
)

CONTRASTIVE_CLAUSE = (
    "\nA REFERENCE GALLERY is attached before the query. USE IT TO DECIDE, BUT NEVER "
    "MENTION IT.\n"
    "The gallery exists only while you write. The model trained on your output will be "
    "shown the query plot ALONE -- no gallery, no references. So a sentence like "
    "'resembles the gallery's X references' teaches it to cite something that will not "
    "exist, and it is also circular: asserting resemblance to the correct class's "
    "examples is only possible because you were given the answer.\n"
    "Therefore:\n"
    "- Do NOT write 'gallery', 'reference', 'references', 'examples', 'as shown above', "
    "or 'resembles the ... examples'. Not once.\n"
    "- DO express the discrimination as a self-contained, checkable test on THIS "
    "recording: name the feature, where it occurs, and why it rules out the most "
    "confusable class. Good: 'the QRS is broad with discordant ST-T in leads 4-6, which "
    "separates conduction disturbance from hypertrophy, where high voltage occurs "
    "without QRS widening.' Bad: 'matches the conduction disturbance references.'\n"
    "- Lead with the OBSERVATION, never with the conclusion.\n"
    "- Keep concrete anchors: sample indices, lead/channel names, and approximate "
    "amplitudes in units. Do not trade a measurement away for a comparison.\n"
)


# ---------------------------------------------------------------------------
# Substance descriptions (SmellNet text_data/text_description.json)
# ---------------------------------------------------------------------------

_SENSOR_RE = None
_VOLATILE_RE = None


def _sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def compact_description(text: str) -> str:
    """Keep the two sentences that bear on reading a sensor plot; drop the botany.

    The upstream descriptions are ~750 characters each and mostly cultivation and
    culinary context. Two kinds of sentence are actually predictive here:

      * the VOLATILE-COMPOUND sentence ("dominated by eugenol", "benzaldehyde",
        "sulfur compounds") -- these map onto which MOX channels should respond;
      * the SENSING sentence ("In gas sensor analysis, cloves typically generate
        high-intensity, long-lasting signals with sharp early peaks") -- a direct
        prior on the response SHAPE, which is exactly what the plot shows.

    Keeping only these cuts 50 substances from ~9.4k tokens to ~3k while retaining
    the part that could change a prediction. Falls back to the final sentence, which
    in this corpus is a summary, when neither pattern is present (8 of 50).
    """
    import re

    global _SENSOR_RE, _VOLATILE_RE
    if _SENSOR_RE is None:
        _SENSOR_RE = re.compile(
            r"\bIn (gas sensor analysis|olfactory sensing|sensor analysis|gas sensing|"
            r"sensing|electronic nose)", re.I
        )
        _VOLATILE_RE = re.compile(
            r"\b(volatile|compound|aroma compound|eugenol|benzaldehyde|terpene|sulfur|"
            r"ester|aldehyde|phenol|pinene|limonene|thiol|allicin)", re.I
        )
    sents = _sentences(text)
    sensor = [s for s in sents if _SENSOR_RE.search(s)]
    volatile = [s for s in sents if _VOLATILE_RE.search(s) and s not in sensor]
    picked = (volatile[:1] + sensor[:1]) or sents[-1:]
    return " ".join(picked)


def load_descriptions(path: Path, mode: str = "compact") -> dict[str, str]:
    """Load substance descriptions keyed by label. mode: compact | full | none."""
    if mode == "none" or not path or not Path(path).is_file():
        return {}
    raw = json.loads(Path(path).read_text())
    if mode == "full":
        return {k: v.strip() for k, v in raw.items()}
    return {k: compact_description(v) for k, v in raw.items()}


def descriptions_block(descs: dict[str, str], labels: list[str] | None = None) -> str:
    """Render descriptions as a system-prompt section, restricted to `labels`."""
    keys = [k for k in (labels or sorted(descs)) if k in descs]
    if not keys:
        return ""
    lines = [
        "SUBSTANCE PRIORS. For each candidate substance, its dominant volatile "
        "chemistry and the gas-sensor response it typically produces. Use these to "
        "decide which channels should respond and with what shape; they are priors, "
        "not guarantees, and the plot overrides them where they disagree.",
    ]
    for k in sorted(keys):
        lines.append(f"- {k}: {descs[k]}")
    return "\n".join(lines) + "\n"
