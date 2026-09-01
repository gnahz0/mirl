#!/usr/bin/env python3
"""Render the time-series datasets (ecg / haptic_ts) into static
graph images and rewrite their annotation JSONs to reference the images instead
of the raw signal files. (Ported from the historical fork's baseline branch.)

Why: the combined Qwen3-VL run consumes images/videos, not raw `.pt`/`.csv`
time series. The "time series as images" approach plots each signal once and
points the sample at that PNG (replacing the `<time-series>` token with
`<image>`), so the existing vision pipeline can ingest them.

Subcommands (run in order):
  render   -- plot every unique signal -> /scratch/<user>/ts_images/<ds>/<hash>.png
              (parallel; writes <ds>_map.json mapping signal_path -> image_path)
  rewrite  -- per record: images=[{"image": png}], drop `signals`, and fix the
              prompt so the number of <image> tokens == number of images (1).
  verify   -- assert every referenced image exists, is non-empty, and that the
              <image> token count matches images per record.

Idempotent: render skips existing PNGs; rewrite is a no-op on records that
already have no `signals`.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mirl_ext.data.schema import DATA_ROOT, SCRATCH_ROOT  # noqa: E402

DATA = DATA_ROOT
IMG_ROOT = os.path.join(SCRATCH_ROOT, "ts_images")
DATASETS = ("ecg", "haptic_ts")

# Unified render quality across all datasets (kept just under Qwen3-VL's
# ~1.0M-pixel processor cap so nothing gets downscaled and token cost is even).
RENDER_DPI = 110


def _img_path(dataset, signal_path):
    h = hashlib.md5(signal_path.encode()).hexdigest()[:20]
    return os.path.join(IMG_ROOT, dataset, f"{h}.png")


def _render_lines(series, ylabels, *, figsize, lw, color, ylabel_fontsize, xlabel, title, out):
    """Shared stacked-line-panel skeleton for ecg. Keeps the Agg
    preamble in-body: pool workers may hit this as the first pyplot import."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(series), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, y, label in zip(axes, series, ylabels):
        ax.plot(y, lw=lw, color=color)
        ax.set_ylabel(label, fontsize=ylabel_fontsize)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=RENDER_DPI)
    plt.close(fig)



def _render_ecg(pt_path, out):
    import torch

    a = torch.load(pt_path, map_location="cpu", weights_only=False).float().numpy()
    if a.ndim == 1:
        a = a[None, :]
    C = a.shape[0]
    _render_lines(
        list(a),
        [f"lead {i + 1}" for i in range(C)],
        figsize=(9, max(3, 1.1 * C)),
        lw=0.9,
        color="k",
        ylabel_fontsize=8,
        xlabel="sample",
        title="ECG leads",
        out=out,
    )


def _render_haptic(pt_path, key, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    o = torch.load(pt_path, map_location="cpu", weights_only=False)
    tac_maps = o.get("tactile") or {}
    if key not in tac_maps:
        key = (o.get("tactile_keys") or list(tac_maps.keys()) or [None])[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [2, 1]})
    if key is not None and key in tac_maps:
        tac = tac_maps[key].float().numpy()
        T = tac.shape[0]
        flat = tac.reshape(T, -1).T
        im = axes[0].imshow(flat, aspect="auto", cmap="viridis", origin="lower")
        axes[0].set_title(f"Tactile taxel activation over time ({key})")
        axes[0].set_ylabel("taxel")
        axes[0].set_xlabel("frame")
        fig.colorbar(im, ax=axes[0], fraction=0.03)
    force = o.get("hand_force_stats")
    if force is not None:
        f = force.float().numpy()
        axes[1].plot(f[:, 0], lw=1.2)
        cols = o.get("hand_force_columns") or []
        axes[1].set_title("Total force over time" + (f" ({cols[0]})" if cols else ""))
        axes[1].set_xlabel("frame")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=RENDER_DPI)
    plt.close(fig)


def _render_one(task):
    dataset, signal_path, key, out, force = task
    try:
        if not force and os.path.exists(out) and os.path.getsize(out) > 0:
            return (signal_path, out, "skip", "")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.png"
        if dataset == "ecg":
            _render_ecg(signal_path, tmp)
        elif dataset == "haptic_ts":
            _render_haptic(signal_path, key, tmp)
        os.replace(tmp, out)
        return (signal_path, out, "ok", "")
    except Exception as e:
        return (signal_path, out, "error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")


def _signals_of(rec):
    return [(s.get("signal"), s.get("key")) for s in (rec.get("signals") or []) if isinstance(s, dict) and s.get("signal")]


def _files(dataset):
    return [os.path.join(DATA, f"{dataset}_{sp}.json") for sp in ("train", "valid")]


def _collect_unique(dataset):
    seen = {}
    for f in _files(dataset):
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for sp, key in _signals_of(json.loads(line)):
                    seen.setdefault(sp, key)
    return seen


def cmd_render(args):
    for ds in (args.datasets or list(DATASETS)):
        uniq = _collect_unique(ds)
        tasks = [(ds, sp, key, _img_path(ds, sp), args.force) for sp, key in uniq.items()]
        print(f"[{ds}] {len(tasks)} unique signals -> rendering with {args.workers} workers", flush=True)
        mapping, ok, skip, err, errors = {}, 0, 0, 0, []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_render_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                sp, out, status, msg = fut.result()
                done += 1
                if status == "error":
                    err += 1
                    if len(errors) < 5:
                        errors.append((sp, msg))
                else:
                    mapping[sp] = out
                    ok += status == "ok"
                    skip += status == "skip"
                if done % 2000 == 0 or done == len(tasks):
                    print(f"  [{ds}] {done}/{len(tasks)} ok={ok} skip={skip} err={err}", flush=True)
        os.makedirs(IMG_ROOT, exist_ok=True)
        map_path = os.path.join(IMG_ROOT, f"{ds}_map.json")
        with open(map_path, "w") as fh:
            json.dump(mapping, fh)
        print(f"[{ds}] DONE ok={ok} skip={skip} err={err} mapped={len(mapping)} -> {map_path}", flush=True)
        for sp, msg in errors:
            print(f"    ERR {sp}: {msg}", flush=True)


def _fix_tokens(content, n_images):
    if "<time-series>" in content:
        content = re.sub(r"(<time-series>\s*)+", "<image>\n", content, count=1)
        content = content.replace("<time-series>", "")
    cnt = content.count("<image>")
    if cnt == n_images:
        return content
    if cnt > n_images:
        parts = content.split("<image>")
        return "<image>".join(parts[: n_images + 1]) + "".join(parts[n_images + 1:])
    return ("<image>\n" * (n_images - cnt)) + content


def cmd_rewrite(args):
    for ds in (args.datasets or list(DATASETS)):
        mapping = json.load(open(os.path.join(IMG_ROOT, f"{ds}_map.json")))
        for f in _files(ds):
            if not os.path.exists(f):
                continue
            n = changed = missing_map = 0
            fd, tmp = tempfile.mkstemp(dir=DATA, suffix=".tmp")
            with os.fdopen(fd, "w") as out, open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n += 1
                    sigs = _signals_of(rec)
                    if sigs:
                        imgs, ok = [], True
                        for sp, _ in sigs:
                            ip = mapping.get(sp)
                            if not ip:
                                ok = False
                                break
                            imgs.append({"image": ip})
                        if not ok:
                            missing_map += 1
                        else:
                            rec["images"] = imgs
                            rec.pop("signals", None)
                            for m in rec.get("prompt", []):
                                if m.get("role") == "user":
                                    m["content"] = _fix_tokens(m["content"], len(imgs))
                            changed += 1
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if missing_map:
                os.unlink(tmp)
                raise SystemExit(f"[{ds}] {os.path.basename(f)}: {missing_map} unmapped signals; run render first")
            os.replace(tmp, f)
            print(f"[{ds}] {os.path.basename(f)}: records={n} converted={changed}", flush=True)


def cmd_verify(args):
    grand_ok = grand_bad = 0
    for ds in (args.datasets or list(DATASETS)):
        for f in _files(ds):
            if not os.path.exists(f):
                continue
            n = bad_img = tok_mismatch = still_sig = 0
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n += 1
                    if rec.get("signals"):
                        still_sig += 1
                    imgs = [im.get("image") if isinstance(im, dict) else im for im in (rec.get("images") or [])]
                    for ip in imgs:
                        if not (ip and os.path.exists(ip) and os.path.getsize(ip) > 0):
                            bad_img += 1
                    utext = "".join(m["content"] for m in rec.get("prompt", []) if m.get("role") == "user")
                    if utext.count("<image>") != len(imgs):
                        tok_mismatch += 1
            good = bad_img == 0 and tok_mismatch == 0 and still_sig == 0
            grand_ok += good
            grand_bad += not good
            print(f"[{ds}] {os.path.basename(f)}: records={n} bad_images={bad_img} token_mismatch={tok_mismatch} leftover_signals={still_sig} [{'OK' if good else 'PROBLEM'}]")
    print(f"\nfiles OK={grand_ok} PROBLEM={grand_bad}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("render", cmd_render), ("rewrite", cmd_rewrite), ("verify", cmd_verify)):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="*", default=None, choices=DATASETS)
        if name == "render":
            p.add_argument("--workers", type=int, default=32)
            p.add_argument("--force", action="store_true", help="re-render even if PNG already exists")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
