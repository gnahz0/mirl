"""Sanity-check Stage 1 alignment training: did weights actually move?

Loads:
  * the saved trainable_visual state-dict at step_500 and step_final
  * the exact original Qwen3.5 model.visual weights

For each pair, computes per-parameter L2 distance and the global cosine of
"how much we moved away from the init", per layer-block. If the deltas are
~0 across the board, we didn't actually train; if they're meaningful but
not enormous, training did its job.

Also dumps:
  * proj head weight norms (small but should be non-zero)
  * log_logit_scale value
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_ckpt(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def load_original_visual_state(model_path: str) -> dict:
    """Load only the native Qwen3.5 vision tensors, without the 9B language model."""
    from mirl_ext.alignment.model import _load_exact_qwen35_visual

    model = _load_exact_qwen35_visual(
        model_path,
        dtype=torch.bfloat16,
        attn_impl="sdpa",
    )
    visual_sd = model.state_dict()
    del model
    return visual_sd


def diff_state_dicts(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    label: str,
    max_print: int = 10,
) -> dict:
    """Compute per-tensor L2 distance + relative change."""
    common = sorted(set(a.keys()) & set(b.keys()))
    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))

    total_l2 = 0.0
    total_norm_a = 0.0
    n_changed = 0
    n_zero = 0
    per_tensor: list[tuple[str, float, float, float]] = []  # (name, l2, ||a||, rel)

    for k in common:
        ta = a[k].float()
        tb = b[k].float()
        if ta.shape != tb.shape:
            print(f"  shape mismatch on {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        delta = (ta - tb).norm().item()
        norm_a = ta.norm().item()
        rel = delta / max(norm_a, 1e-12)
        total_l2 += delta ** 2
        total_norm_a += norm_a ** 2
        if delta == 0.0:
            n_zero += 1
        else:
            n_changed += 1
        per_tensor.append((k, delta, norm_a, rel))

    print(f"\n=== diff: {label} ===")
    print(f"  shared params: {len(common)} (changed={n_changed}, zero-delta={n_zero})")
    if only_a:
        print(f"  only in A: {len(only_a)} (e.g. {only_a[:3]})")
    if only_b:
        print(f"  only in B: {len(only_b)} (e.g. {only_b[:3]})")
    global_l2 = total_l2 ** 0.5
    global_norm = total_norm_a ** 0.5
    global_rel = global_l2 / max(global_norm, 1e-12)
    print(f"  global L2(B-A) = {global_l2:.4f}")
    print(f"  global ||A||    = {global_norm:.4f}")
    print(f"  global relative change = {global_rel:.6f}  ({global_rel*100:.4f}%)")

    per_tensor.sort(key=lambda x: x[3], reverse=True)
    print(f"  top {max_print} tensors by relative change:")
    for name, delta, na, rel in per_tensor[:max_print]:
        print(f"    {rel*100:7.4f}%  ||delta||={delta:.5f}  ||a||={na:.5f}  {name}")

    return {
        "n_shared": len(common),
        "n_changed": n_changed,
        "n_zero_delta": n_zero,
        "global_l2": global_l2,
        "global_norm_a": global_norm,
        "global_rel": global_rel,
        "top_changed": [
            {"name": n, "l2": d, "norm_a": na, "rel": r}
            for n, d, na, r in per_tensor[:max_print]
        ],
    }


def report_projection_heads(state: dict) -> None:
    print("\n=== projection heads + log_logit_scale ===")
    for key in ("proj_visual", "proj_text"):
        if key not in state:
            print(f"  {key:<14}  MISSING")
            continue
        sd = state[key]
        norms = [(n, p.float().norm().item()) for n, p in sd.items()]
        total = sum(n**2 for _, n in norms) ** 0.5
        print(f"  {key:<14}  ||W||_global={total:.4f}  ({len(norms)} tensors)")
    lls = state.get("log_logit_scale")
    if lls is not None:
        v = float(lls.float().mean().item())
        print(f"  log_logit_scale = {v:.4f}  (scale = exp(.) = {torch.tensor(v).exp().item():.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-root",
        default="/scratch/dvdai_mit/alecz/checkpoints/alignment_qwen35_siglip2_unified_grayscale_clean_v1",
    )
    ap.add_argument(
        "--qwen-path",
        default="/work/mit/ppliang_mit/alecz/hf_cache/hub/models--Qwen--Qwen3.5-9B/"
        "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    ap.add_argument("--skip-original", action="store_true",
                    help="Skip comparison vs HF original (slow; needs net or local cache).")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    early = root / "step_500" / "alignment_state.pt"
    final = root / "final" / "alignment_state.pt"

    if not early.exists():
        # Fall back to step_10 if step_500 doesn't exist.
        early = root / "step_10" / "alignment_state.pt"

    print(f"loading early ckpt: {early}")
    s_early = load_ckpt(early)
    print(f"loading final ckpt: {final}")
    s_final = load_ckpt(final)
    print(f"early keys: {sorted(s_early.keys())}")

    # ---- diff between early and final on the trainable visual encoder ----
    diff_state_dicts(
        s_early["trainable_visual"],
        s_final["trainable_visual"],
        label=f"early({early.parent.name}).trainable_visual  vs  final.trainable_visual",
    )

    report_projection_heads(s_final)

    # ---- diff between final and the exact Qwen3.5 model.visual init ----
    if not args.skip_original:
        print(f"\nloading original Qwen3.5 model.visual from {args.qwen_path} ...")
        try:
            orig_visual = load_original_visual_state(args.qwen_path)
            diff_state_dicts(
                orig_visual,
                s_final["trainable_visual"],
                label="ORIGINAL Qwen3.5 model.visual vs final.trainable_visual",
            )
        except Exception as e:
            print(f"  could not load original: {e!r}")

    # Quick read of wandb summary, if accessible
    wandb_dir = root / "wandb" / "wandb"
    if wandb_dir.exists():
        latest_runs = sorted(wandb_dir.glob("run-*"))
        if latest_runs:
            last_run = latest_runs[-1]
            summary = last_run / "files" / "wandb-summary.json"
            if summary.exists():
                print(f"\n=== wandb summary ({last_run.name}) ===")
                with open(summary) as f:
                    j = json.load(f)
                # Print just the loss-ish stuff
                for k in sorted(j.keys()):
                    if any(s in k for s in ("loss", "lr", "logit", "n/", "skipped")):
                        v = j[k]
                        print(f"  {k:<24}  {v}")


if __name__ == "__main__":
    main()
