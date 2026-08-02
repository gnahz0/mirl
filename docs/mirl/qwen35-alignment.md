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

Smell sensors and heterogeneous force fields use robust normalization. ECG can
use the same historical transform or `prestandardized`, a linear
`clip(x, -4, 4) / 4` mapping for the stored per-lead z-scored tensors. The tactile
pressure cube uses one recording-level robust scale so relative pressure across
its physical 16×16 surface is not erased:

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
- Put each 32-step window in one video frame. A channel occupies one 32×32 merger
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
- Keep native temporal resolution through 256 frames, which fully preserves 76.5%
  of recordings. Uniformly cover the full duration only for the longer tail. This
  retains 70.0% of all source frames overall, versus 24.9% under the former
  64-frame cap.
- Normalize the tactile cube at recording level to preserve relative spatial
  pressure; normalize the heterogeneous force-summary fields independently.
- With `tactile_delta_channels=true`, encode pressure in R/B and the normalized
  frame delta in G; otherwise repeat pressure across RGB. Qwen's native temporal
  convolution still fuses adjacent frames.
- Nearest-neighbor expand each 16×16 tactile map to one 32×32 merger cell.
- Place the 13 aligned right-force summaries in an adjacent 32×32 cell.
- Qwen's temporal kernel fuses adjacent frame pairs.
- Preserve signed v2 calibration values; do not clamp negatives.

An `F`-frame clip produces `2 × ceil(F/2)` output tokens: two spatial tokens
(tactile and force) per temporal frame pair, with an odd final frame repeated as
Qwen requires. Training uses one 21-frame peak-centered tactile crop; full
validation clips retain at most 256 frames. The smoke cap is 64 frames.

## Training recipe

- Each rank receives 4 smell, 4 ECG, 4 tactile, and 20 real image/video samples.
  This is a 32-sample microbatch per rank. Two-step gradient accumulation gives
  an effective global batch of 256 on four AICR B200s without activation checkpointing.
- SmellNet sampling and metrics use only the 50 base substances; mixture rows
  never enter the active dataset or W&B tables.
- The clean baseline aligns each Qwen-encoded sensor crop directly with its frozen
  SigLIP2 label prototype. GC-MS is disabled; it can be enabled later as a separate
  ablation without changing the baseline result.
- Training crops are 100 smell steps, 750 ECG steps (3 seconds at 250 Hz), and
  21 tactile frames. There is no separate crop-consistency objective.
- The visual/text projections are linear into 512 dimensions. Effective dimension
  is diagnostic only; the baseline has no variance or covariance objective.
- AdamW uses `1e-5` for Qwen and projection heads, `3e-3` for the
  temperature, weight decay `0.05`, gradient clipping `1.0`, and 5% warmup.
- The production schedule is 1,500 optimizer steps. Validation runs every 100
  steps over five deterministic balanced batches, visiting all 50 base SmellNet
  validation recordings once.
- Checkpoint selection focuses on SmellNet sensor-to-text top-1 at
  `val-core/accuracy/smellnet`. Accuracy and macro-F1 are also logged for ECG,
  haptic, and the equal-family overall; per-class tables remain under `val-aux/`.
