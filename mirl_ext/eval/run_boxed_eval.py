"""Offline greedy \\boxed{} eval over the GRPO val parquets (base vs SFT arms).

Mirrors the RL serving path end to end: rows go through MIRLDataset (same
message building, media caps, and overlong-prompt filter as
run_qwen35_grpo.sh), prompts render with the model's chat template via verl's
wrapper, media is extracted with the same qwen_vl_utils path the agent loop
uses, and vLLM receives the same unexpanded-placeholder prompt + raw media the
rollout engine gets. Outputs are scored with the SAME per-family scorer RL
uses (mirl_ext.rewards.combined).

    python -m mirl_ext.eval.run_boxed_eval --model /path/to/hf_model \\
        --data-root "$MIRL_DATA_ROOT" --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from mirl_ext.data.dataset import MIRLDataset
from mirl_ext.data.schema import DATA_ROOT
from mirl_ext.rewards import combined
from mirl_ext.rewards._common import extract_boxed_answer
from verl.utils.tokenizer.chat_template import apply_chat_template

VAL_FAMILIES = [
    "ecg_valid",
    "haptic_ts_valid",
    "climb_valid",
    "human_behaviour_valid_fast",
    "tactile_valid_fast",
]

# Requests per llm.generate call: decoded media for a whole family must never
# sit in RAM at once (ecg_valid is ~10k image rows).
GEN_CHUNK = 512


def _data_config(args) -> DictConfig:
    """The data.* contract from run_qwen35_grpo.sh (media caps + RL's val filter)."""
    return OmegaConf.create(
        {
            "image_patch_size": 16,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": args.filter_workers,
            "truncation": "error",
            "max_video_frames": args.max_video_frames,
            "max_video_bytes": 52428800,
            "max_image_tokens": 12288,
            "max_image_tokens_total": 24576,
            "cache_dir": os.path.join(os.environ.get("TMPDIR", "/tmp"), "verl-rlhf-cache"),
        }
    )


def _to_request(row: dict, processor, config, patch_size: int) -> dict:
    """One vLLM request, ordered as the agent loop is: process_multi_modal_info
    mutates the messages (caps/drops), then the template renders those same
    messages. vLLM re-expands the placeholders from the attached media, exactly
    like the rollout engine (verl dedups back to single pads before generate)."""
    messages = row["raw_prompt"]
    images, videos, audios = MIRLDataset._process_multi_modal_info(messages, patch_size, config)
    prompt = apply_chat_template(processor, messages, tokenize=False)
    mm = {key: val for key, val in (("image", images), ("video", videos), ("audio", audios)) if val}
    return {"prompt": prompt, "multi_modal_data": mm} if mm else {"prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None, help="processor dir if the model export lacks one")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--families", nargs="*", default=VAL_FAMILIES)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--max-samples", type=int, default=0, help="per family, pre-filter; 0 = all")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--max-video-frames",
        type=int,
        default=24,
        help="safety ceiling; tactile uses 1 FPS (4..24), while human and CLIMB request 8",
    )
    ap.add_argument("--max-model-len", type=int, default=15360)
    ap.add_argument("--max-prompt-length", type=int, default=11264, help="RL's data.max_prompt_length filter")
    ap.add_argument("--filter-workers", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true", help="build datasets + requests and check scorer routes, then exit before vLLM")
    args = ap.parse_args()

    paths = {fam: Path(args.data_root) / f"{fam}.parquet" for fam in args.families}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing val parquets: {missing}")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    config = _data_config(args)

    if args.dry_run:
        for family, path in paths.items():
            dataset = MIRLDataset(str(path), processor.tokenizer, config, processor, max_samples=args.max_samples or -1)
            rows = [dataset[i] for i in range(len(dataset))]
            for row in rows:
                assert isinstance(row["extra_info"], dict), f"{family}: extra_info not decoded to dict"
                combined.compute_score(row["data_source"], "", str(row["reward_model"]["ground_truth"]))  # route exists
            requests = [_to_request(row, processor, config, dataset.image_patch_size) for row in rows]
            with_media = sum("multi_modal_data" in req for req in requests)
            print(f"dry-run {family:28s} rows={len(rows)} media={with_media}")
        print("dry-run OK (stopped before vLLM init)")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 8, "video": 1},  # measured val max: 4 images / 1 video per row
        seed=0,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    responses_path = args.out.with_suffix(".responses.jsonl")
    responses_path.write_text("")
    results = {
        "model": args.model,
        "data_root": str(args.data_root),
        "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "max_prompt_length": args.max_prompt_length,
        "families": {},
    }

    print(f"\n{'family':28s} {'n':>5s} {'rows':>5s} {'acc':>6s} {'boxed':>6s} {'major':>6s} {'margin':>7s} {'score':>6s}")
    for family, path in paths.items():
        file_rows = pq.ParquetFile(path).metadata.num_rows
        dataset = MIRLDataset(str(path), processor.tokenizer, config, processor, max_samples=args.max_samples or -1)
        rows = [dataset[i] for i in range(len(dataset))]
        if not rows:
            print(f"{family:28s} {0:5d} {file_rows:5d}  (all rows filtered out)")
            continue
        outs = []
        for start in range(0, len(rows), GEN_CHUNK):
            chunk = rows[start : start + GEN_CHUNK]
            # Media decode/resize releases the GIL; threads mirror the agent loop's executor offload.
            with ThreadPoolExecutor(max_workers=8) as pool:
                requests = list(pool.map(lambda r: _to_request(r, processor, config, dataset.image_patch_size), chunk))
            outs.extend(llm.generate(requests, sampling))

        gts = [str(r["reward_model"]["ground_truth"]) for r in rows]
        acc_sum = boxed_sum = score_sum = 0.0
        per_src: dict[str, dict] = {}
        with responses_path.open("a") as fh:
            for row, gt, out in zip(rows, gts, outs):
                text = out.outputs[0].text
                metrics = combined.compute_score(row["data_source"], text, gt)
                boxed = extract_boxed_answer(text)
                acc_sum += metrics["acc"]
                score_sum += metrics["score"]
                boxed_sum += boxed is not None
                stats = per_src.setdefault(row["data_source"], {"n": 0, "acc": 0.0, "score": 0.0, "boxed": 0.0, "gts": Counter()})
                stats["n"] += 1
                stats["acc"] += metrics["acc"]
                stats["score"] += metrics["score"]
                stats["boxed"] += boxed is not None
                stats["gts"][gt] += 1
                fh.write(
                    json.dumps(
                        {
                            "family": family,
                            "data_source": row["data_source"],
                            "ground_truth": gt,
                            "boxed": boxed,
                            "acc": metrics["acc"],
                            "score": metrics["score"],
                            "response": text,
                        }
                    )
                    + "\n"
                )
        n = len(rows)
        acc = acc_sum / n
        majority = max(Counter(gts).values()) / n
        results["families"][family] = {
            "n": n,
            "file_rows": file_rows,
            "accuracy": acc,
            "boxed_rate": boxed_sum / n,
            "mean_score": score_sum / n,
            "majority_baseline": majority,
            "margin_over_majority": acc - majority,
            "per_source": {
                src: {
                    "n": s["n"],
                    "accuracy": s["acc"] / s["n"],
                    "mean_score": s["score"] / s["n"],
                    "boxed_rate": s["boxed"] / s["n"],
                    "majority_baseline": max(s["gts"].values()) / s["n"],
                }
                for src, s in sorted(per_src.items())
            },
        }
        print(
            f"{family:28s} {n:5d} {file_rows:5d} {acc:6.3f} {boxed_sum / n:6.3f} "
            f"{majority:6.3f} {acc - majority:+7.3f} {score_sum / n:6.3f}"
        )
        args.out.write_text(json.dumps(results, indent=2))

    if "ecg_valid" in results["families"]:
        print(
            "note: ECG acc is single-label top-1 via category-mention matching; ECG is "
            "inherently multi-label (majority ~0.44) -- a proxy, prefer macro metrics offline."
        )
    print(f"-> {args.out}\n-> {responses_path}")


if __name__ == "__main__":
    main()
