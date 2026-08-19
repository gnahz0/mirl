# MIRL SFT (cold-start) plan

Goal: add a **supervised fine-tuning cold-start** stage before GRPO, on data held
**disjoint** from the RL set, so RL never sees SFT examples (avoids leakage /
memorization inflating RL reward). This mirrors the Revisual-R1 recipe already cited
in the README (optimized cold-start → staged RL).

Pipeline: **split 50:50 → generate CoT targets (GPT-5.x) → veRL SFT → GRPO from the SFT ckpt on the RL half.**

---

## 1. Data split (50:50, SFT ⟂ RL)

RL data today (veRL format, one row = one example):
```
data_source, prompt[<image>\n…question…], images[{image: /…/ts_images/…png}], videos, reward_model{style, ground_truth}, extra_info
```
Families: `ecg (39,449)`, `smellnet`, `haptic_ts`, `climb`, `human_behaviour`, `tactile`.

Split rules:
- **Stratified per `data_source`** (and per class within it where cheap), random, **fixed seed=42**, exactly 50/50. Stratifying keeps both halves class-balanced instead of letting a random split skew rare classes.
- Output layout (new, does not touch originals):
  ```
  data/split/rl/<family>_train.parquet     # RL half — becomes GRPO's train_files
  data/split/sft/<family>_train.parquet     # SFT half — input to target generation
  data/split/split_manifest.json            # row ids per half, seed, counts (reproducible)
  ```
- **Val sets stay shared/untouched** — both stages evaluate on the same held-out val.
- Script: `mirl_ext/sft/split_sft_rl.py` (reads each family parquet, stratified shuffle, writes both halves + manifest).

## 2. SFT target generation (uses the Microsoft GPT-5.x credits)

SFT needs `(prompt → response)`; we have prompt + `ground_truth` but **no response**. Generate one via
**answer-conditioned rationalization** (STaR-style): give GPT the question + option list + the *correct* answer,
ask for a completion in the exact RL reward format:
```
<think> {concise, signal-plausible reasoning that lands on the answer} </think> \boxed{ {ground_truth} }
```
Why this format: the reward (`mirl_ext/rewards/*.py`) scores `re.fullmatch(<think>.*</think>.*\boxed{.*})` and the answer inside `\boxed{}`. Matching it means SFT teaches exactly the format+reasoning scaffold GRPO rewards, so rollouts start diverse (the README notes format-forced `<think>` is what kept non-collapsing tasks alive).

**Honest caveat:** GPT does **not** see the pseudo-image raster (it's a non-interpretable grayscale signal encoding), so the CoT is a *plausible rationalization conditioned on the label*, not grounded perception. That's fine for cold-start — it teaches output format + reasoning shape; **RL then grounds it** in the actual VE features. Sending the PNG to GPT-vision would waste tokens for no gain.

Design:
- `mirl_ext/sft/gen_sft_targets.py`
  - OpenAI client → `base_url=http://point.dd.works:18890/v1`, `api_key` read at runtime from `~/.config/mirl/microsoft_openai_key` (never hardcoded, never logged).
  - **Model fallback ladder** on rate-limit / 429 / empty: `gpt-5.5_2026-04-24` → `gpt-5.1_2025-11-13`. (`gpt-5.3-chat_2026-03-03` verified dead — 404 DEPLOYMENT_NOT_FOUND — skip it.)
  - **Use `max_completion_tokens`**, NOT `max_tokens` (these models reject the latter). Endpoint + key smoke-tested OK 2026-07-26.
  - Concurrency (async or thread pool ~8–16), exponential backoff, **checkpoint-to-jsonl so it resumes** (never re-bills a completed row).
  - **Validate each completion** against the reward regex + that `\boxed{}` equals the ground_truth; drop/retry non-conforming ones (rejection sampling on format).
  - Runs **locally or on the login node** (needs outbound net) — only pulls the *text* columns, not the PNGs.
- **Scale/cost:** the SFT half is ~84k rows — you do **not** need all of them. Cold-start SFT is typically a balanced ~10–20k CoT subset. Start with a per-family cap (e.g. 2–3k each) and a hard budget; scale up only if RL warm-start underperforms. Log exactly how many were dropped for non-conformance.

## 3. Build SFT parquet

`mirl_ext/sft/build_sft_parquet.py`: join validated completions back to their prompt+image rows and emit veRL SFT format
(`prompt`/`messages` + `response`, image path preserved) → `data/split/sft/<family>_sft.parquet`.

## 4. Train SFT (veRL's built-in engine — no new trainer)

Reuse `tests/special_e2e/sft/run_sft_engine.sh` / `examples/sft/*` (FSDP, sequence packing, Liger `USE_LIGER=1`, optional LoRA):
- base = the Qwen3.5-9B path (same as GRPO's `MODEL_PATH`); multimodal image inputs.
- launcher `mirl_ext/sft/run_sft_b200.sbatch`.
- output → `/scratch/dvdai_mit/alecz/checkpoints/sft_qwen35_v1`.
- Short: 1–3 epochs, low LR (~1e-5), cosine, bf16 — standard cold-start.

## 5. GRPO from the SFT checkpoint

Point `examples/mirl/multiverse/run_qwen35_grpo.sh` `MODEL_PATH` at the SFT checkpoint and set `train_files` to the
**RL half** (`data/split/rl/*`). Everything else unchanged.

---

## Files to create
| Path | Purpose |
|---|---|
| `mirl_ext/sft/split_sft_rl.py` | stratified 50:50 split + manifest |
| `mirl_ext/sft/gen_sft_targets.py` | GPT-5.x CoT generation (key from file, fallback ladder, resume) |
| `mirl_ext/sft/build_sft_parquet.py` | validated completions → veRL SFT parquet |
| `mirl_ext/sft/run_sft_b200.sbatch` | SFT launcher |
| `docs/mirl/sft-plan.md` | this file |

## Open decisions (need your call)
1. **SFT size:** all ~84k, or a capped balanced subset (recommended ~10–20k)?
2. **Split granularity:** per-family only, or per-class within family (better for rare ECG classes)?
3. **Reasoning length:** short 1–3 sentence `<think>` (cheaper, less overfit) vs. longer CoT?
4. **LoRA vs full SFT** for the 9B cold-start.
