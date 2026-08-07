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
  normalization is fixed and the shared temperature remains trainable. Each
  family derives its bias from `1 / number_of_classes`.
- **Preservation loss:** sample-balanced cosine distance between every
  pre-merger student and anchor encoder token on real images/videos.

Transformers computes SigLIP2 loss only inside the full model forward and assumes
a square one-image/one-text identity pairing. The sensor objective instead has a
rectangular sample-by-label matrix, repeated positives, and class-balanced rows,
so it uses PyTorch's equivalent `binary_cross_entropy_with_logits` primitive
directly. Cosine preservation uses `cosine_similarity` and `segment_reduce`;
there is no project-local loss module.

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

A semantic tile boundary must therefore be a multiple of 32 pixels. Stage 1
mean-pools the model's 1152-D pre-merger `last_hidden_state` for sensor alignment
and preserves every pre-merger image/video token. The frozen 4096-D merger remains
the interface consumed by the language tower in Stage 2.

The SigLIP2 text tower is the closest contrastive label encoder because the Qwen
vision architecture/initialization derives from SigLIP2-SO400M. Qwen exposes its
1152-D pre-merger encoder state, which matches SigLIP2's 1152-D text output, so
the normalized features are compared directly with no projection head. The
pretrained 4096-D Qwen merger remains frozen.

## Full raw-data audit

| family | rows | on-disk contract | important variation |
|---|---:|---|---|
| SmellNet | 2,279 | CSV | 1,979 mixture files: timestamp + 4 sensors; 300 base files: 6 sensors. Native lengths 517–900. |
| ECG | 49,305 | bare contiguous `torch.bfloat16` tensor, `format=ts_pt` | Every file is 8×2500 across PTB-XL (21,837), Georgia (10,344), Chapman-Shaoxing (10,247), and CPSC (6,877). |
| tactile | 2,210 | `torch.float32` dictionary, `format=tactile_pt` | v1 (1,318): right 16×16 tactile only; v2 (892): left/right/aligned/mat tactile. The loader selects the right tactile tensor. |

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

Some tactile files also contain force statistics and additional v2 maps. The
clean baseline ignores that optional metadata and loads only the selected right
16×16 pressure sequence, giving every tactile sample the same tensor contract.

## Temporal representation

The families share a `[-1, 1]` numeric contract. Smell sensor rows and tactile
taxels use robust normalization over time. ECG uses `clip(x, -4, 4) / 4`
because the stored tensors are already per-lead z-scored.

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

### Tactile: pseudo-video

- Every source is a 30-FPS sequence of 16×16 pressure maps (median 177 frames;
  range 47–4,606).
- Feed every recording at its complete native duration. There is no temporal
  windowing, cropping, frame selection, or cap.
- Robustly normalize each taxel over the full recording and repeat pressure
  identically across RGB. There is no fixed pressure ceiling or engineered
  force/delta channel.
- Nearest-neighbor expand each 16×16 tactile map to one 32×32 merger cell.
- Qwen's temporal kernel fuses adjacent frame pairs.

A recording with `T` tactile frames produces `ceil(T / 2)` output tokens: one
spatial pressure token per temporal frame pair. The median 177-frame recording
therefore produces 89 tokens; the longest observed 4,606-frame recording
produces 2,303 tokens.

## Training recipe

- Every microbatch contains one media kind and one `data_source`; SigLIP negatives
  still come from the complete family label bank.
- Stage 1 expands multi-image or multi-video rows into one preservation anchor per
  unique physical path because it ignores QA annotations. SFT and RL must instead
  preserve row grouping and load all media in prompt order.
- Visual rows and signal sources omitted from `train.signal_repeat_factors` are
  visited once per epoch. Configured low-resource signal sources repeat complete
  independently shuffled passes. Validation is always one-pass. Source groups
  smaller than the GPU count are skipped so every rank receives a sample.
- SmellNet sampling and metrics use only the 50 base substances; mixture rows
  never enter the active dataset or W&B tables.
- The clean baseline aligns every sensor family through one complete SigLIP2
  text-label bank per split: 50 SmellNet labels, 7 ECG labels, and one annotated
  open answer per tactile recording. Filename-derived task stems are not used;
  stored label casing is preserved, and each answer is truncated to SigLIP2's
  64-token context and encoded once.
- SmellNet, ECG, and tactile all retain their complete native time axes in both
  training and validation. One dataset row produces one sensor embedding.
- Frozen SigLIP2 embeddings and Qwen's 1152-D pre-merger states are compared
  directly.
- Accelerate wraps the model with standard DDP. Sensor loss is evaluated on
  rank-local rows against the complete label bank; only class counts and metric
  statistics are reduced across ranks.
- AdamW uses `1e-5` for Qwen, `3e-3` for the
  temperature, weight decay `0.05`, gradient clipping `1.0`, and 3% warmup.
- The production schedule is one sampler epoch. Validation runs every 200 steps
  over the complete one-pass sensor validation sampler, visiting every SmellNet,
  ECG, and tactile validation recording once.
- Every signal family logs accuracy, macro-F1, Recall@1, Recall@5, and mAP.
  `overall` is their equal-family mean across SmellNet, ECG, and tactile;
  checkpoint selection uses `val-core/f1_macro/overall`. Tactile Recall@1/5 and
  mAP remain the primary interpretation because its 635 captions are unique.
