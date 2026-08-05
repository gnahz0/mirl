# Qwen3.5 time-series vision alignment

## Model topology

- **Student VE:** exact `model.visual.*` tensors from the installed
  `Qwen/Qwen3.5-9B` snapshot; trainable.
- **Anchor VE:** a frozen copy of that same loaded tower.
- **Label encoder:** frozen text tower from
  `google/siglip2-so400m-patch16-naflex`.
- **Time-series loss:** class-balanced **SigLIP sigmoid**
  ([arXiv:2303.15343](https://arxiv.org/abs/2303.15343)) against each family's
  complete fixed label vocabulary. SigLIP2 text features are cached once; their
  projection and the shared temperature remain trainable. Each family derives
  its bias from `1 / number_of_classes`.
- **Preservation loss:** sample-balanced cosine distance between every
  post-merger student and anchor token on real images/videos.

The Qwen tower is not replaced by Google's standalone vision tower. Qwen3.5
uses a Qwen-specific `Qwen3_5VisionModel` whose exact final weights are loaded
from the Qwen checkpoint. Its geometry is:

| field | value |
|---|---:|
| vision parameters | 456,010,480 |
| layers / hidden / heads | 27 / 1152 / 16 |
| spatial patch | 16×16 |
| temporal patch | 2 frames |
| spatial merger | 2×2 patches |
| post-merger output | 4096 |

A semantic tile boundary must therefore be a multiple of 32 pixels. The model's
`last_hidden_state` is pre-merger 1152-D; alignment uses the 4096-D
`pooler_output`, whose token sequence is what the language tower consumes.
Stage 1 mean-pools each time-series sequence for classification, while image and
video preservation operates on the full post-merger token sequence.

The SigLIP2 text tower is the closest paired contrastive label encoder because
the Qwen vision architecture/initialization derives from SigLIP2-SO400M.
Qwen subsequently vision-language-trained its tower, so learned visual/text
projection heads are retained rather than assuming the final spaces are
identical.

## Full raw-data audit

| family | rows | on-disk contract | important variation |
|---|---:|---|---|
| SmellNet | 2,279 | CSV | 1,979 mixture files: timestamp + 4 sensors; 300 base files: 6 sensors. Native lengths 517–900. |
| ECG | 49,305 | bare contiguous `torch.bfloat16` tensor, `format=ts_pt` | Every file is 8×2500 across PTB-XL (21,837), Georgia (10,344), Chapman-Shaoxing (10,247), and CPSC (6,877). |
| haptic | 2,210 | `torch.float32` dictionary, `format=tactile_pt` | v1 (1,318): right 16×16 tactile only; v2 (892): left/right/aligned/mat tactile. Right tactile is selected consistently. |

The cleaned SmellNet indexes contain 1,192 training recordings (250 base and
942 mixture) and 297 validation recordings (50 base and 247 mixture) after
removing byte/numerical duplicates under different paths. Stage 1 currently
excludes `smellnet_mixture` before it constructs label vocabularies, samplers, or
metrics. The active task is therefore the 50-way base benchmark: five training
recordings and one validation recording for every substance. There are zero
shared paths, basenames, raw hashes, or numerical hashes across the splits.

ECG source-specific exception: four Georgia files contained whole missing leads,
for 12,500 NaNs total:

- `E07941.pt`: lead 7
- `E06909.pt`: lead 3
- `E04603.pt`: leads 0 and 1
- `E07675.pt`: lead 4

All four occur in the training split and are excluded by
`ecg_train_nan_filtered.parquet` (39,449 → 39,445 rows). The matching validation
index is named `ecg_valid_nan_filtered.parquet` and remains at 9,856 rows because
none occur there. The raw `.pt` files are retained on scratch for provenance and
recovery; neither alignment config can sample them.

Haptic files also contain aligned `hand_force_stats`: 13 right-force summaries
for v1 and 26 left+right summaries for v2. The loader selects the 13 `right_*`
columns to match the right tactile map and maintain a common v1/v2 schema. One
v1 record has no force statistics and receives a masked force tile. v2-only
left/aligned/mat maps remain an explicit future ablation rather than appearing
only for a subset of records.

## Temporal representation

The families share a `[-1, 1]` numeric contract, not one statistical transform.
Smell sensors and heterogeneous force fields use robust normalization. ECG uses
`clip(x, -4, 4) / 4` because the stored tensors are already per-lead z-scored.
Tactile pressure follows OpenTouch's fixed physical scale: clip raw pressure to
`[0, 3072]`, divide by 3072, then map the result to `[-1, 1]`.

Robustly normalized rows use:

`scale = 0.7 * (MAD / 0.6745) + 0.3 * std`

`pixel = tanh((value - median) / (2 * scale))`

MAD is weighted over std (0.7/0.3) because outlier resistance is the whole point
of using MAD; an even split re-injects the std's spike sensitivity. The tanh gain
of 2 (not 4) uses more of the `[-1, 1]` range — a ±1σ sample maps to
`tanh(0.5) ≈ 0.46` rather than `tanh(0.25) ≈ 0.245` — so the pseudo-video has real
contrast while tanh still saturates genuine outliers.

This follows TimeOmni-VL's robust-fidelity idea and maps directly to Qwen's
normal `[-1, 1]` pixel range without line plots, PIL, or uint8 quantization.
The normalized scalar intensity is repeated identically across RGB. Exact `-1`
marks missing/padded pixels; no color channel has a separate engineered meaning.

### Unified scalar video: SmellNet and ECG

- Treat both inputs identically as `(channels, time)`; only their native dimensions
  differ.
- Put each consecutive 32-step block in one video frame. A channel occupies one 32×32 merger
  cell, with its 32 values repeated vertically and identically across RGB.
- Qwen's native temporal patcher fuses adjacent frames, so each output token covers
  64 ordered timesteps from exactly one sensor or lead.
- Keep native samples: no plots, period estimator, interpolation, or downsampling.
- ECG periodicity is retained in the ordered pixel values for the trainable tower
  to learn; the input converter does not estimate beats or introduce an ECG-only
  layout prior.

Example token counts:

| shape | post-merger tokens |
|---|---:|
| SmellNet base: 6×867 | 84 |
| ECG: 8×2500 | 320 |

All families use this temporal Qwen input grammar; there is no image/hybrid mode.

### Haptic: pseudo-video

- Every source is a 30-FPS sequence of 16×16 pressure maps (median 177 frames;
  range 47–4,606).
- Feed every recording at its complete native duration. There is no temporal
  windowing, cropping, frame selection, or cap.
- Normalize pressure with OpenTouch's fixed 3072-count scale. Normalize the
  heterogeneous force-summary fields independently within each recording.
- With `tactile_delta_channels=true`, encode pressure in R/B and the normalized
  frame delta in G; otherwise repeat pressure across RGB. Qwen's native temporal
  convolution still fuses adjacent frames.
- Nearest-neighbor expand each 16×16 tactile map to one 32×32 merger cell.
- Place the 13 aligned right-force summaries in an adjacent 32×32 cell.
- Qwen's temporal kernel fuses adjacent frame pairs.

A recording with `T` tactile frames produces `2 * ceil(T / 2)` output tokens:
two spatial tokens (tactile and force) per temporal frame pair. The median
177-frame recording therefore produces 178 tokens; the 4,606-frame maximum
produces 4,606 tokens.

## Training recipe

- Each rank receives up to 8 SmellNet recordings, 8 ECG recordings, 8 tactile
  recordings, and 8 real image/video samples. Two-step gradient accumulation
  gives an effective global recording batch of 256 on four AICR B200s.
- One epoch consumes every sensor recording once. A family disappears from later
  batches after it is exhausted; it is never recycled to match a larger family or
  the auxiliary image pool. Images are sampled without replacement alongside the
  sensor-defined epoch.
- SmellNet sampling and metrics use only the 50 base substances; mixture rows
  never enter the active dataset or W&B tables.
- The clean baseline aligns SmellNet and ECG recordings with frozen SigLIP2 class
  prototypes. Each tactile recording is paired with the complete annotated answer;
  filename-derived task stems are not used. Answers longer than SigLIP2's 64-token
  context produce multiple equally weighted positive chunks. Chunks are pooled
  only when ranking complete answers for retrieval. GC-MS remains a separate ablation.
- SmellNet, ECG, and tactile all retain their complete native time axes in both
  training and validation. One dataset row produces one sensor embedding.
- The visual/text projections are linear into 512 dimensions. Effective dimension
  is diagnostic only; the baseline has no variance or covariance objective.
- AdamW uses `1e-5` for Qwen and projection heads, `3e-3` for the
  temperature, weight decay `0.05`, gradient clipping `1.0`, and 3% warmup.
- The production schedule is one sensor epoch. Validation runs every 100 steps
  over the complete one-pass sensor validation sampler, visiting every SmellNet,
  ECG, and haptic validation recording once.
- Checkpoint selection uses the SmellNet/ECG validation macro-F1 at
  `val-core/f1_macro/overall`. Haptic logs bidirectional sensor/text Recall@1,
  Recall@5, and mAP over all 635 validation recordings.
