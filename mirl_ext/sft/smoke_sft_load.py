"""One-row smoke test of the ACTUAL veRL SFT path per built parquet.

For every file: instantiate the pinned MultiTurnSFTDataset with the real
Qwen3.5 tokenizer+processor, tokenize row 0, collate a 2-row batch, and check
(a) prompt tokens carry no loss and the assistant completion does, (b) media
placeholder counts match the media arrays (the dataset hard-asserts this),
(c) the tokenized SFT prompt is a prefix of the serve-time GRPO/inference
render (train/serve consistency), and (d) a true leading system turn really
does break this path -- the documented reason sft_messages() merges it.
--forward additionally runs one model forward pass (GPU node).

Run on the cluster inside srun (never the login node):
    srun -p cpu -c 4 --mem=32G <env>/bin/python mirl_ext/sft/smoke_sft_load.py \\
        --parquets .../sft_parquet/*_sft.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirl_ext.sft.export_sft_tasks import _config_path  # noqa: E402

QWEN35_PATH = _config_path("cluster_qwen35_path", "MIRL_QWEN35_PATH", "")


def check_one(path: Path, tokenizer, processor, cfg, forward_model=None) -> bool:
    import torch
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    from verl.utils.dataset.dataset_utils import SFTTensorCollator

    ds = MultiTurnSFTDataset(parquet_files=str(path), tokenizer=tokenizer, config=cfg,
                             processor=processor)
    item = ds[0]
    input_ids, loss_mask = item["input_ids"], item["loss_mask"]
    n_loss = int(loss_mask.sum())
    assert n_loss > 0, "no supervised tokens"
    assert int(loss_mask[: (loss_mask == 1).nonzero()[0, 0]].sum()) == 0, "loss on prompt tokens"
    supervised = tokenizer.decode(input_ids[loss_mask.bool()])
    assert "\\boxed{" in supervised and "</think>" in supervised, (
        f"supervised span is not the completion: {supervised[:200]!r}"
    )

    row = ds.dataframe.iloc[0].to_dict()
    messages = [dict(m) for m in row["messages"]]
    text = "".join(m["content"] for m in messages)
    n_img, n_vid = text.count("<image>"), text.count("<video>")
    assert n_img == len(row.get("images") or []), "image placeholder/media mismatch"
    assert n_vid == len(row.get("videos") or []), "video placeholder/media mismatch"

    # Train/serve: the serve-time render (prompt + generation prompt) must be a
    # prefix of the SFT sequence; report the first divergence otherwise.
    serve_ids = tokenizer.apply_chat_template(
        [{k: m[k] for k in ("role", "content")} for m in messages[:-1]],
        add_generation_prompt=True, tokenize=True,
    )
    flat = input_ids.tolist()
    match_len = next(
        (i for i, (a, b) in enumerate(zip(serve_ids, flat)) if a != b), min(len(serve_ids), len(flat))
    )
    ts_ok = match_len == len(serve_ids)
    ts_note = "" if ts_ok else (
        f" TRAIN/SERVE DIVERGES at token {match_len}: "
        f"serve...{tokenizer.decode(serve_ids[max(0, match_len - 8): match_len + 4])!r} vs "
        f"sft...{tokenizer.decode(flat[max(0, match_len - 8): match_len + 4])!r}"
    )

    batch = SFTTensorCollator(cfg["pad_mode"])([ds[i] for i in range(min(2, len(ds)))])
    mm_keys = sorted((item.get("multi_modal_inputs") or {}).keys())
    print(f"[OK] {path.name}: rows={len(ds)} seq={len(flat)} loss_tokens={n_loss} "
          f"mm={mm_keys} collated={type(batch).__name__} serve_prefix={'ok' if ts_ok else 'MISMATCH'}"
          + ts_note)

    if forward_model is not None:
        inputs = {"input_ids": input_ids.unsqueeze(0).to(forward_model.device),
                  "attention_mask": torch.ones(1, len(flat), dtype=torch.long,
                                               device=forward_model.device)}
        for k, v in (item.get("multi_modal_inputs") or {}).items():
            inputs[k] = v.to(forward_model.device)
        with torch.no_grad():
            out = forward_model(**inputs, use_cache=False)
        print(f"     forward: logits {tuple(out.logits.shape)}")
    return ts_ok


def check_system_turn_breaks(tokenizer, processor, cfg, sample_parquet: Path) -> None:
    """Demonstrate why sft_messages() merges the system turn (focused test)."""
    import json
    import tempfile

    import pandas as pd

    df = pd.read_parquet(sample_parquet).head(1).copy()
    msgs = [dict(m) for m in df.iloc[0]["messages"]]
    df.at[df.index[0], "messages"] = [{"role": "system", "content": "You are an expert."}] + msgs
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name)
        try:
            from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

            MultiTurnSFTDataset(parquet_files=tmp.name, tokenizer=tokenizer,
                                config=cfg, processor=processor)[0]
            print("[??] a leading system turn tokenized cleanly -- re-evaluate the "
                  "sft_messages() merge (and delete this note if intentional)")
        except Exception as exc:  # noqa: BLE001
            print(f"[OK] leading system turn breaks the pinned SFT path as documented: "
                  f"{type(exc).__name__}: {str(exc)[:140]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquets", nargs="+", required=True, type=Path)
    ap.add_argument("--model-path", default=QWEN35_PATH or None, required=not QWEN35_PATH)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--pad-mode", default="no_padding", choices=["no_padding", "right"])
    ap.add_argument("--forward", action="store_true", help="also run one model forward (GPU)")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    processor = hf_processor(args.model_path, trust_remote_code=True)
    if processor is not None and processor.chat_template is None:
        processor.chat_template = tokenizer.chat_template
    cfg = OmegaConf.create({
        "pad_mode": args.pad_mode, "max_length": args.max_length, "truncation": "error",
        "messages_key": "messages", "image_key": "images", "video_key": "videos",
    })

    model = None
    if args.forward:
        import torch
        from transformers import AutoModelForCausalLM

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for cls in (AutoVLM, AutoModelForCausalLM):
            try:
                model = cls.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                            trust_remote_code=True).to(device).eval()
                break
            except (ValueError, OSError) as exc:
                print(f"[warn] {cls.__name__} could not load the checkpoint: {exc}")
        if model is None:
            raise SystemExit("no Auto class could load the model for --forward")

    all_ok = True
    for path in args.parquets:
        all_ok &= check_one(path, tokenizer, processor, cfg, model)
    check_system_turn_breaks(tokenizer, processor, cfg, args.parquets[0])
    print("\nALL CHECKS PASSED" if all_ok else "\nTRAIN/SERVE MISMATCH -- see lines above")


if __name__ == "__main__":
    main()
