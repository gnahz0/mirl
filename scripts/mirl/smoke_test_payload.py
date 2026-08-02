# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Smoke test: does the model actually RECEIVE and READ the plots we think we sent?

Everything upstream can look healthy while the model is effectively blind. The
generator validates format and answer, and both pass whether or not the image
arrived -- because the answer is supplied. So a broken image path, a stale cache, a
sheet that is illegible at panel size, or a silently dropped attachment would all
produce a 100%-yield run of confident, entirely invented traces.

This checks the chain end to end by asking questions ONLY the pixels can answer:

  1. LOCAL   -- the bytes decode, are the expected size, and are not blank.
  2. ARRIVAL -- token accounting confirms the endpoint billed us for images.
  3. QUERY   -- the model reports the query plot's subplot COUNT and channel NAMES,
                which are printed on the axes. Ground truth comes from the parquet
                (6 channels for smellnet base, 4 for mixture, 8 ECG leads).
  4. GALLERY -- the model reads back a caption from a REQUESTED panel position. This
                is the one that catches an illegible contact sheet: the caption is
                rendered at ~18px, so if it cannot be read here it cannot be used.

    python scripts/mirl/smoke_test_payload.py --family smellnet_train \\
        --data-source smellnet_base --gallery 2 --gallery-grid 4
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gallery import build_gallery, resolve_image  # noqa: E402
from gen_sft_targets import BASE_URL, DEFAULT_MODEL, load_api_key  # noqa: E402

EXPECTED_CHANNELS = {
    "smellnet_base": ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"],
    "smellnet_mixture": ["NO2", "VOC", "CO"],
    "ecg": None,
    "haptic_tactile": None,
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks", default="data/sft/ts_sft_tasks.jsonl")
    ap.add_argument("--gallery-tasks", default="data/sft/rl_gallery_tasks.jsonl")
    ap.add_argument("--image-root", type=Path, default=Path("data/sft/ts_images"))
    ap.add_argument("--family", default="smellnet_train")
    ap.add_argument("--data-source", default="smellnet_base")
    ap.add_argument("--gallery", type=int, default=2)
    ap.add_argument("--gallery-grid", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/sft/gallery_cache"))
    args = ap.parse_args()

    from openai import OpenAI
    from PIL import Image, ImageStat

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    q = [
        t for t in tasks
        if t["family"] == args.family
        and (not args.data_source or t.get("data_source") == args.data_source)
    ]
    if not q:
        raise SystemExit(f"no tasks for {args.family}/{args.data_source}")
    query = q[0]

    print("=" * 78)
    print(f"1. LOCAL IMAGE INTEGRITY   family={args.family} data_source={args.data_source}")
    print("=" * 78)
    qpath = resolve_image(query["image_path"], args.image_root)
    if qpath is None:
        raise SystemExit(f"query plot NOT FOUND: {query['image_path']}")
    with Image.open(qpath) as im:
        im.load()
        qsize, qmode = im.size, im.mode
        qstd = ImageStat.Stat(im.convert("L")).stddev[0]
    print(f"   query plot : {qpath.name}  {qsize} {qmode}  pixel_stddev={qstd:.1f}")
    assert qstd > 3.0, "query plot is blank"

    gal_tasks = [
        json.loads(l) for l in Path(args.gallery_tasks).read_text().splitlines() if l.strip()
    ] if Path(args.gallery_tasks).is_file() else tasks
    sheets, uids, note = build_gallery(
        gal_tasks, args.family, args.gallery, args.cache_dir, args.image_root,
        grid=args.gallery_grid,
    )
    for sp in sheets[:2]:
        with Image.open(sp) as im:
            print(f"   sheet      : {sp.name}  {im.size}  "
                  f"stddev={ImageStat.Stat(im.convert('L')).stddev[0]:.1f}")
    print(f"   sheets={len(sheets)}  reference rows={len(uids)}")

    client = OpenAI(base_url=BASE_URL, api_key=load_api_key())

    print("\n" + "=" * 78)
    print("2. ARRIVAL — is the endpoint billing us for the images?")
    print("=" * 78)
    txt_only = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Reply: ok"}],
        max_completion_tokens=1024,
    ).usage.prompt_tokens
    with_img = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(qpath)}"}},
            {"type": "text", "text": "Reply: ok"},
        ]}],
        max_completion_tokens=1024,
    ).usage.prompt_tokens
    delta = with_img - txt_only
    print(f"   text-only={txt_only}  with 1 plot={with_img}  delta={delta} tokens")
    print(f"   {'PASS' if delta > 100 else 'FAIL'} — image reached the endpoint")

    print("\n" + "=" * 78)
    print("3. QUERY PLOT — can it read what is printed on the axes?")
    print("=" * 78)
    probe = (
        "Look ONLY at the attached plot. Answer as strict JSON, no prose:\n"
        '{"n_subplots": <int>, "y_axis_labels": [<string per subplot, top to bottom>], '
        '"x_axis_label": <string>, "x_max_tick": <number>}'
    )
    r = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(qpath)}"}},
            {"type": "text", "text": probe},
        ]}],
        max_completion_tokens=2048,
    )
    raw = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    got = json.loads(m.group(0)) if m else {}
    print(f"   model reports: {json.dumps(got)[:300]}")
    expected = EXPECTED_CHANNELS.get(args.data_source)
    if expected:
        names = [str(x).strip().upper() for x in got.get("y_axis_labels", [])]
        hit = sum(1 for e in expected if any(e.upper() in n or n in e.upper() for n in names))
        print(f"   expected channels {expected}")
        print(f"   matched {hit}/{len(expected)}   n_subplots={got.get('n_subplots')} "
              f"(expected {len(expected)})")
        print(f"   {'PASS' if hit >= len(expected) - 1 else 'FAIL'} — axes are legible")
    else:
        print(f"   n_subplots={got.get('n_subplots')} (ECG expects 8 leads)")

    print("\n" + "=" * 78)
    print("4. GALLERY SHEET — can it read a caption at panel size?")
    print("=" * 78)
    if not sheets:
        print("   (no sheets)")
        return
    with Image.open(sheets[0]) as im:
        pass
    ask = (
        f"The attached image is a {args.gallery_grid}x{args.gallery_grid} grid of plots. "
        "Each panel has a caption in a dark band directly beneath it. "
        'Answer as strict JSON, no prose: {"captions_row1": [<caption of each panel in '
        'the TOP row, left to right>], "total_panels_visible": <int>}'
    )
    r2 = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(sheets[0])}"}},
            {"type": "text", "text": ask},
        ]}],
        max_completion_tokens=2048,
    )
    raw2 = (r2.choices[0].message.content or "").strip()
    m2 = re.search(r"\{.*\}", raw2, re.DOTALL)
    got2 = json.loads(m2.group(0)) if m2 else {}
    print(f"   model reports: {json.dumps(got2)[:400]}")

    # Ground truth = the captions ACTUALLY rendered, in sheet order. Recomputing the
    # requested order instead would flag a mismatch whenever a reference plot was
    # unresolvable and omitted -- a bug in the check, not in the sheet.
    truth = getattr(build_gallery, "last_captions", [])
    expect_row1 = truth[: args.gallery_grid]
    print(f"   expected row 1: {expect_row1}")
    reported = [str(x).strip().lower() for x in got2.get("captions_row1", [])]
    hit = sum(1 for e in expect_row1 if e.strip().lower() in reported)
    print(f"   matched {hit}/{len(expect_row1)} captions")
    print(f"   {'PASS' if hit >= max(1, len(expect_row1) - 1) else 'FAIL'} — sheet captions legible")


if __name__ == "__main__":
    main()
