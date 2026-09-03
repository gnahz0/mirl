# Qwen3.5 time-series vision alignment

## Model topology

- **Student VE:** exact `model.visual.*` tensors from the installed
  `Qwen/Qwen3.5-9B` snapshot; trainable.
- **Anchor VE:** a frozen copy of that same loaded tower.
- **Label encoder:** frozen text tower from
  `google/siglip2-so400m-patch16-naflex`.
- **Time-series loss:** **SigLIP sigmoid**
  ([arXiv:2303.15343](https://arxiv.org/abs/2303.15343)) against cached SigLIP2
  text labels. ECG uses the complete label vocabulary found in the active split.
  Tactile uses one 30-label bank spanning six closed QA tasks, with multi-positive
  targets for initial-contact fingers and highest-pressure fingers.
- **Preservation loss:** sample-balanced cosine distance between every
  pre-merger student and anchor encoder token on real images.

Transformers computes SigLIP2 loss only inside the full model forward and assumes
a square one-image/one-text identity pairing. The sensor objective instead has
rectangular sample-by-label matrices and repeated positives, so it uses PyTorch's
equivalent `binary_cross_entropy_with_logits` primitive directly. ECG sums each
sample's complete candidate bank before an inverse-frequency class-balanced
sample mean. Tactile masks labels outside the row's observed tasks, averages the
observed pairs within each task, then gives the six observed tasks equal weight.
Cosine preservation uses `cosine_similarity` and `segment_reduce`; there is no
project-local loss module.

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
and preserves every pre-merger image token. The frozen 4096-D merger remains
the interface consumed by the language tower in Stage 2.

The SigLIP2 text tower is the closest contrastive label encoder because the Qwen
vision architecture/initialization derives from SigLIP2-SO400M. Qwen exposes its
1152-D pre-merger encoder state, which matches SigLIP2's 1152-D text output, so
the normalized features are compared directly with no projection head. The
pretrained 4096-D Qwen merger remains frozen.

## Full raw-data audit

| family | rows | on-disk contract | important variation |
|---|---:|---|---|
| ECG | 49,305 | bare contiguous `torch.bfloat16` tensor, `format=ts_pt` | Every file is 8×2500 across PTB-XL (21,837), Georgia (10,344), Chapman-Shaoxing (10,247), and CPSC (6,877). |
| tactile | 2,210 | `torch.float32` dictionary, `format=tactile_pt` | v1 (1,318): right 16×16 tactile only; v2 (892): left/right/aligned/mat tactile. The production index selects a tensor key for each recording (currently the right tactile map). |

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
clean baseline ignores that optional metadata and loads the `[time, height,
width]` pressure sequence named by the row's single `signals` entry, giving every
tactile sample the same tensor contract.

The file names describe two views of the same recordings:
`haptic_ts_{train,valid}.parquet` supplies the native pressure sequences, while
`tactile_{train,valid}.parquet` supplies the closed QA annotations. The latter's
RGB-video rows are both visual-preservation samples and the annotation table
used for the label join.
`AlignmentDataset` constructs train and validation separately and joins the QA
answers to native sequences by recording stem within that split.

## Temporal representation

Both signal families use the same finite-value check and z-score transform:
subtract the mean, divide by a standard deviation clamped to at least `1e-6`,
clip to ±4 standard deviations, and divide by 4. ECG applies it independently
to each lead over time. Tactile applies one global transform to all values in a
recording. A constant signal therefore maps to zero.

The result is a floating-point `[-1, 1]` Qwen video tensor without line plots,
PIL, or uint8 quantization. Scalar intensity is repeated identically across RGB;
no color channel has a separate engineered meaning. ECG padding pixels are
exactly `-1`, after normalization, so they remain distinct from constant data.

### ECG scalar pseudo-video

- Treat the input as `(channels, time)`.
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
| ECG: 8×2500 | 320 |

Every ECG row uses this temporal grammar. Image-preservation rows retain their
native visual inputs instead of being converted to scalar frames.

### Tactile: pseudo-video

- Every source is a 30-FPS sequence of 16×16 pressure maps. In the current
  truncated indexes, training spans 47–720 frames (median 179) and validation
  spans 63–720 frames (median 171).
- Feed every timestep in each referenced tensor. There is no alignment-time
  windowing, frame selection, or maximum-frame setting; the corpus preparation
  step has already truncated the longest recordings to 24 seconds.
- Globally z-score the complete recording and repeat pressure identically across
  RGB. There is no fixed pressure ceiling or engineered force/delta channel.
- Nearest-neighbor expand each 16×16 tactile map to one 32×32 merger cell.
- Qwen's temporal kernel fuses adjacent frame pairs.

A recording with `T` tactile frames produces `ceil(T / 2)` output tokens: one
spatial pressure token per temporal frame pair. The median 179-frame training
recording therefore produces 90 tokens; the current 720-frame maximum produces
360 tokens.

## Training recipe

- Every microbatch contains one media kind and one `data_source`; SigLIP negatives
  still come from the complete family label bank.
- Stage 1 uses the first image path from each image row as a preservation anchor.
  Human-behavior and CLIMB videos use at most eight uniformly spaced frames.
  Tactile RGB+heatmap composites use approximately 1 FPS with a four-frame
  floor and 24-frame ceiling. Native signals are never downsampled by either
  video rule.
- Visual rows and signal sources omitted from `train.signal_repeat_factors` are
  visited once per epoch. Configured low-resource signal sources repeat complete
  independently shuffled passes. Validation is always one-pass. Source groups
  smaller than the GPU count are skipped so every rank receives a sample.
- Configured repeats rebalance the low-resource tactile source without forcing
  the two sensor families to equal exposure. The AICR config repeats
  `haptic_tactile` 5 times per epoch (approximate square-root balancing), as
  does the ORCD config;
  image sources and ECG are one-pass. The six tactile task losses are weighted
  equally.
- ECG aligns to the complete split vocabulary (seven labels in the current
  indexes). Tactile joins six closed QA tasks to pressure recordings by shared
  recording stem: initial fingers, highest-pressure fingers, force level, grip
  stability, contact feature, and local shape. Initial fingers and
  highest-pressure fingers are multi-label; the other four tasks are exactly
  one label. Their 30 choices are verbalized and encoded once. Conflicting
  annotations remove only that task, while a tactile recording with no
  unambiguous closed-task annotation fails dataset construction. Open tactile
  responses are ignored as supervision; tactile video pixels remain visual
  preservation inputs.
- Collation represents each tactile recording with a `[30]` multi-hot target and
  a `[30]` observation mask. An observed task marks its entire label span in the
  mask; positive choices set one or more target bits. Unobserved or conflicting
  tasks stay masked out, so their zero targets are never trained as negatives.
  The shared parser always returns a set of option indices; task specifications
  enforce 1–6 choices for the two multi-label tasks and exactly one choice for
  the other four.
- ECG and tactile retain their complete native time axes in both training and
  validation. One signal row produces one sensor embedding.
- Frozen SigLIP2 embeddings and Qwen's 1152-D pre-merger states are compared
  directly.
- Accelerate wraps the model with standard DDP. Sensor loss is evaluated on
  rank-local rows; only small class/observation counts and metric statistics are
  reduced across ranks.
- AdamW applies one configured learning rate to all trainable parameters
  (`1e-5` on AICR and `5e-5` on ORCD), with zero weight decay, gradient clipping
  at `1.0`, and 3% warmup. The learned logit scale and bias share this optimizer.
- Both production configs run three sampler epochs. Validation runs every 200
  optimizer steps and at each epoch end over the complete one-pass validation
  sampler, visiting every ECG and tactile recording once.
- `train.init_checkpoint` names an `alignment_state.pt` file and warm-starts the
  encoder and learned temperature with a fresh optimizer and schedule.
  `train.resume_checkpoint` names the matching `last/trainer_state.pt` file and
  additionally restores optimizer and scheduler state; it requires
  `train.init_checkpoint` to point at the sibling `last/alignment_state.pt`.
  Validation overwrites `last/` at optimizer-step boundaries while `best/` and
  `final/` remain lightweight model exports.
- Every signal family logs accuracy, Recall@1, Recall@5, and mAP, plus auxiliary
  prediction-coverage statistics. Accuracy is an any-positive top-1 hit;
  Recall@k is the fraction of a sample's positive labels recovered in the top k.
  They coincide at k=1 for single-label tasks but not for multi-positive tactile
  tasks. Tactile first averages its six tasks equally, then `overall` averages
  ECG and tactile equally. Checkpoint selection uses `val-core/map/overall`.
