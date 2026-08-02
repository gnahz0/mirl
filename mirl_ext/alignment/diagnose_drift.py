# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.
"""Report Qwen vision-tower parameter drift from its pristine checkpoint.

    srun -p cpu -c 8 --mem=64G --time=00:30:00 python -m mirl_ext.alignment.diagnose_drift \\
        --ckpt /scratch/dvdai_mit/alecz/checkpoints/<run>/best/alignment_state.pt

"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _tier(name: str) -> str:
    if name.startswith("patch_embed"):
        return "patch_embed"
    if name.startswith("merger"):
        return "merger"
    if "attn" in name:
        return "attn"
    if "mlp" in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def _block_index(name: str) -> int | None:
    parts = name.split(".")
    if parts and parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint directory or alignment_state.pt")
    ap.add_argument(
        "--config",
        default=None,
        help="config YAML (default: checkpoint sibling, then the production config)",
    )
    ap.add_argument("--top", type=int, default=12, help="show N most-moved tensors")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from mirl_ext.alignment.model import _load_exact_qwen35_visual

    ckpt_path = Path(args.ckpt)
    config_path = Path(args.config) if args.config else Path("mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml")
    if args.config is None:
        candidate = (ckpt_path if ckpt_path.is_dir() else ckpt_path.parent) / "config.yaml"
        if candidate.is_file():
            config_path = candidate
    cfg = OmegaConf.load(config_path)
    ckpt_file = ckpt_path / "alignment_state.pt" if ckpt_path.is_dir() else ckpt_path
    state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    trained = state["trainable_visual"]
    print(f"checkpoint: {ckpt_file}  (step={state.get('step')})")
    print(f"config: {config_path}")
    if "log_logit_scale" in state:
        value = float(state["log_logit_scale"])
        print(f"  log_logit_scale = {value:+.4f} -> temperature {torch.tensor(value).exp():.3f}")
    else:
        print("  log_logit_scale = <ABSENT from checkpoint>")

    print("\nloading pristine Qwen3.5 vision tower for comparison...")
    pristine = _load_exact_qwen35_visual(str(cfg.model.qwen35_path), dtype=torch.float32, attn_impl="sdpa").state_dict()

    rows, exact_zero, missing = [], [], []
    num, den = 0.0, 0.0
    for name, w0 in pristine.items():
        if name not in trained:
            missing.append(name)
            continue
        w = trained[name].to(torch.float32)
        w0 = w0.to(torch.float32)
        d = float((w - w0).norm())
        n0 = float(w0.norm())
        num += d**2
        den += n0**2
        rel = d / n0 if n0 > 0 else 0.0
        if d == 0.0:
            exact_zero.append(name)
        rows.append((rel, d, name))

    global_rel = (num**0.5) / (den**0.5) if den > 0 else 0.0
    print(f"\n=== GLOBAL relative drift  ||W-W0|| / ||W0|| = {global_rel:.3e} ===")
    print(
        f"    tensors compared: {len(rows)}   exactly-zero delta: {len(exact_zero)}   missing from ckpt: {len(missing)}"
    )
    if global_rel < 1e-5:
        print("    VERDICT: encoder is effectively FROZEN -- gradients are not reaching it.")
    elif global_rel < 1e-3:
        print("    VERDICT: encoder moved only marginally (expected very early in warmup).")
    else:
        print("    VERDICT: encoder is training.")
    if exact_zero:
        print(
            f"    WARNING: {len(exact_zero)} tensors have EXACTLY zero delta, e.g. "
            f"{exact_zero[:3]} -- these never received an update."
        )

    by_tier: dict[str, list[float]] = {}
    by_block: dict[int, list[float]] = {}
    for rel, _d, name in rows:
        by_tier.setdefault(_tier(name), []).append(rel)
        b = _block_index(name)
        if b is not None:
            by_block.setdefault(b, []).append(rel)

    print("\n=== drift by component (mean relative) ===")
    for tier, vals in sorted(by_tier.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {tier:12s} n={len(vals):4d}  mean={sum(vals) / len(vals):.3e}  max={max(vals):.3e}")

    if by_block:
        idxs = sorted(by_block)
        print("\n=== drift by block depth (early -> late) ===")
        for b in idxs[:3] + (["..."] if len(idxs) > 6 else []) + idxs[-3:]:
            if b == "...":
                print("  ...")
                continue
            vals = by_block[b]
            print(f"  block {b:2d}  mean={sum(vals) / len(vals):.3e}")

    print(f"\n=== top {args.top} most-moved tensors ===")
    for rel, d, name in sorted(rows, reverse=True)[: args.top]:
        print(f"  {rel:.3e}  (abs {d:.4f})  {name}")

    for head in ("proj_visual", "proj_text"):
        if head in state:
            tot = sum(float(t.float().norm()) ** 2 for t in state[head].values()) ** 0.5
            print(f"\n{head} weight norm = {tot:.4f} (random-init; large is expected)")


if __name__ == "__main__":
    main()
