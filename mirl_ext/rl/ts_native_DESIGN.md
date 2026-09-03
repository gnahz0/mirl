# Native time-series input for RL (`_tsnative` parquets)

Goal: feed the gradable ts family (ecg) to GRPO the way
Stage-1 alignment feeds them — `mirl_ext/data/signals.py` pseudo-video frames —
instead of the current static matplotlib plots (`render_timeseries_images.py`).
The builder also renders haptic_ts strips, but nothing consumes them in RL
(100% open free-text, ungradable — excluded); they are reserved for the future
native-signal SFT path.
Prompts, ground truth, `reward_model`, and `extra_info` stay byte-identical
(only `<image>` becomes `<video>` and the media entry changes), so the scorer
(`rewards/combined.py`) is untouched.

## Representation

Each signal is rendered by the **same signals.py functions Stage-1 uses** (no
duplicated math): `timeseries_frames(signal, cell=32)` for ecg
(`[C,T]` → `ceil(T/32)` frames of `C·32 × 32`, z-scored per channel, ±4σ →
[-1,1], -1 tail padding, channel boundaries on 32 px) and
`tactile_frames(t, side=32)` for haptic_ts — both **identical to Stage-1**.
The frames are quantized to uint8 via `u = round((x+1)·127.5)` and stacked
**vertically into one grayscale PNG per signal**: `<md5-20>_stack{T}.png`
(md5 stem = the row's existing plot-PNG hash, so plot and native variants stay
joinable). One file per signal keeps the inode count at ~54k (vs ~4M for
per-frame PNGs — ECG alone is 49,305 signals × 80 frames).

At load time `MIRLDataset._process_multi_modal_info` detects the
`_stack{T}.png` suffix and fetches it **itself** into the `(frames-tensor,
metadata)` tuple that both consumers (trainer processor call and agent-loop →
vLLM) already pass through for real videos. qwen_vl_utils is deliberately
bypassed for strips: its frame-list path hands `image_factor` (32) to
`fetch_image` as the *patch size*, so frames get smart_resized with factor 64
and the 32 px tile width doubles (measured: `(20,3,192,32)` → `(20,3,192,64)`).
The HF `Qwen3VLVideoProcessor`, by contrast, is measured identity on our dims:
its `smart_resize` counts **whole-video** pixels (`4096 ≤ t·h·w ≤ 25165824`,
factor 32), which every family satisfies — including 32×32 tactile.

**Why the pixel round-trip is exact:** Qwen3.5's processors use
`image_mean = image_std = 0.5`, so rescale+normalize is `(u/255 − 0.5)/0.5` —
the exact inverse of our quantization. Net: the model sees the `[-1,1]`
tensors Stage-1 fed with `do_resize/do_rescale/do_normalize=False`, up to
8-bit quantization (measured max abs diff 1/255 = 0.00392, grids equal). The
builder's `--probe` mode measures this end-to-end rather than trusting the
derivation.

One deviation from Stage-1, counted per row by the builder: a **frame cap**
(default 256, evenly spaced, first+last kept). Haptic recordings run
T=108..3346 and uncapped tails would blow `MAX_PROMPT_LENGTH=11264` with
timestamp runs. Normalization happens **before** the pick, over the full
recording, exactly as Stage-1. ECG (79 frames) never hits
the cap. `data.max_video_frames=8` never touches strips (that knob only feeds
qwen_vl_utils path-video sampling).

Token cost drops vs the plots: ECG ≈ 320 video tokens + 40 timestamp runs
(plots were ~1M px ≈ 4k image tokens).

## Data plumbing

Signals were stripped from the GRPO parquets by the plot rewrite, but
`$DATA/trainedve_raw/<family>_{train,valid}.parquet` (persistent, on /work)
holds the same row populations with `signals` intact — verified 100%
row-index-aligned on ground_truth and data_source across all four files. The
builder joins **by row index** and asserts ground-truth equality per row (the
SFT join guard). Rows whose signal is non-finite (`normalize` raises) are
dropped and counted — expect exactly 4 in ecg_train (the `nan_filtered` delta).

## What changes where

- `mirl_ext/rl/build_ts_native_parquet.py` (new): renders strips to
  `$MIRL_SCRATCH_ROOT/data/ts_native/<family>/` (idempotent, tmp+rename) and
writes `<family>_{train,valid}_tsnative.parquet` next to the originals with
  the **identical Arrow schema** (all four current parquets share
  `videos: list<struct<video, min_frames, max_frames>>`; we fill
  `{"video": strip, "min_frames": None, "max_frames": None}` — the None-popping
  in `MIRLDataset._build_messages` already handles that shape). `--probe` mode
  renders 3 rows/family, saves human-inspectable PNGs, and runs the processor
  equivalence check.
- `mirl_ext/data/dataset.py`: `fetch_ts_stack` plus the strip branch in
  `_process_multi_modal_info` (runs on both the trainer path and the agent-loop
  rollout path — they share the classmethod; `_split_videos_and_metadata` in
  `verl/utils/tokenizer/tokenizer.py` already unpacks the tuple and sets
  `do_sample_frames=False`).
- `examples/mirl/multiverse/run_qwen35_grpo.sh`: `TS_NATIVE=1` selects the
  `_tsnative` files for ecg under the RL-half/closed
  protocol; haptic_ts is excluded from RL entirely.

For the record, `TS_TOKENS=1`/`_tstok` was the third, dormant representation:
the historical raw-numeric-**text** A/B (signals downsampled and printed into
the prompt as numbers, no media at all; see `data/TIME_SERIES_TOKENS.md` on the
cluster). Its pre-built parquets are root-level old-cluster artifacts; the
launcher now errors on `TS_TOKENS=1` — no model adapter, no builder here.

## Open questions

- **Timestamps**: Qwen3.5 interleaves per-temporal-group timestamp text
  (metadata fps defaults to 2.0, so ECG reads as a "40 s video"). Harmless and
  uniform across rows; could later set `sample_fps` to the true sensor rate —
  that changes prompt text, i.e. a new lineage.
- **vLLM-side reprocessing** is derived, not yet measured: the probe covers
  the MIRLDataset fetch + HF video processor; the first smoke run should
  confirm vLLM applies the same processor path (the tactile-mp4 rows already
  validated frames-tensor+metadata flow end-to-end in job 172783).

## Interleaved sparse-video + dense-ts (planned; user decision 2026-09-01)

The thesis-grade experiment: sample VIDEO sparsely (~1 fps or sparser) and the
tactile TIME SERIES densely (native ~30 Hz), temporally interleaved in one
prompt. If sparse-video+dense-ts matches or beats dense-video at equal/lower
token cost, that demonstrates the ts channel is a useful AND efficient input —
the project's central claim.

- Sampling ratio: start 1:1 (one video frame per ts second), allow 1:2+ (one
  video frame anchoring multiple ts steps). Interleave in temporal order.
- Token economics (measured): one ts frame (32x32 taxel map) = 1 merged token;
  one video frame ~40-80 tokens. Dense ts is ~30-80x cheaper per temporal
  sample. Median 6 s grasp at 1 fps video + 30 Hz ts ~ 570 tokens; the 154 s
  compliance sessions need a video cap (~16-24 frames evenly spaced) + full ts
  (~2.3k tokens) to stay inside the 11264 prompt budget.
- Recording lengths (n=300): 66% are 3-8 s single grasps; 8.3% exceed 1000
  frames — all compliance/deformability sessions where squeeze rhythm is the
  signal. Subsample ts only above ~1024 frames; never linspace-truncate.
- Teacher traces use the same format: interleaved image sequence
  [vframe, ts-chunk, vframe, ts-chunk, ...] to the API with a system note
  defining the alignment; ts chunks must go as an ordered sequence (frames,
  never one tall stacked strip).
- Student engineering: true interleaving needs custom media assembly in
  MIRLDataset (alternating video-frame images and ts-chunk pseudo-video
  segments); v0 fallback is two time-aligned streams (sparse video + full ts
  strip) with the alignment stated in the prompt.
- Eval arms to justify the claim: (a) video-8 baseline (current), (b) video
  dense, (c) sparse video + dense ts interleaved, (d) ts only. Token-matched
  comparisons between (b) and (c) carry the argument.
