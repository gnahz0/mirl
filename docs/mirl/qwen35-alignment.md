# Qwen3.5 time-series vision alignment

## Model topology

- **Student VE:** exact `model.visual.*` tensors from the installed
  `Qwen/Qwen3.5-9B` snapshot; trainable.
- **Anchor VE:** a frozen copy of that same loaded tower.
- **Label encoder:** frozen text tower from
  `google/siglip2-so400m-patch16-naflex`.
- **Time-series loss:** duplicate-label-aware symmetric InfoNCE between learned
  projections of the student VE and label embeddings.
- **Preservation loss:** cosine distance between raw, post-merger student and
  anchor features on real images/videos.

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
`pooler_output`, which is what the language tower consumes.

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

The active SmellNet indexes are content-cleaned versions of that raw population.
Train has 1,192 unique numerical recordings after removing 790 byte/numerical
duplicates under different `training_seen`/`training_new` paths; validation has
297 unique recordings. There are zero shared paths, basenames, raw hashes, or
numerical hashes across the splits. Both contain the same six-channel base and
four-channel mixture schemas. Label text is canonicalized to lowercase natural
spacing so capitalization variants of the same mixture are not false negatives.

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

## Hybrid patch representation

All signal values use robust normalization. Smell sensors, ECG leads, and
heterogeneous force fields are scaled independently. The tactile pressure cube
uses one recording-level scale so relative pressure across its physical 16×16
surface is not erased:

`scale = 0.5 * (MAD / 0.6745) + 0.5 * std`

`pixel = tanh((value - median) / (4 * scale))`

This follows TimeOmni-VL's robust-fidelity idea and maps directly to Qwen's
normal `[-1, 1]` pixel range without line plots, PIL, or uint8 quantization.
The normalized scalar intensity is repeated identically across RGB. Exact `-1`
marks missing/padded pixels; no color channel has a separate engineered meaning.

The default `model.ts_representation=hybrid` uses:

### Unified scalar raster: SmellNet and ECG

- Treat both inputs identically as `(channels, time)`; only their native dimensions
  differ.
- Fold each channel's samples through a 32px-high serpentine raster. Adjacent
  columns reverse direction so the one-dimensional temporal path stays spatially
  continuous.
- Each timestep is one pixel: no line plot, interpolation, or downsampling.
- Each sensor/lead owns complete 32×32 merger cells, so merged tokens never cross
  channel boundaries.
- Every block holds at most 1,024 timesteps from one channel. Longer sequences
  expand horizontally by another complete block rather than being resized.
- Within a block, Qwen embeds four 16×16 patches and its 2×2 MLP merger produces
  one 4096-D token. Those four patches therefore contain only one sensor/lead.
- ECG periodicity is retained in the ordered pixel values for the trainable tower
  to learn; the input converter does not estimate beats or introduce an ECG-only
  layout prior.

Example token counts:

| shape | post-merger tokens |
|---|---:|
| SmellNet mixture: 4×600 | 4 |
| SmellNet base: 6×867 | 6 |
| ECG: 8×2500 | 24 |

The explicit `ts_representation=video` ablation also uses the same frequency-free
32-step windows for ECG and SmellNet; there is no period estimator in either path.

### Haptic: pseudo-video

- Every source is a 30-FPS sequence of 16×16 pressure maps (median 177 frames;
  range 47–4,606).
- Keep native temporal resolution through 256 frames, which fully preserves 76.5%
  of recordings. Uniformly cover the full duration only for the longer tail. This
  retains 70.0% of all source frames overall, versus 24.9% under the former
  64-frame cap.
- Normalize the tactile cube at recording level to preserve relative spatial
  pressure; normalize the heterogeneous force-summary fields independently.
- Repeat normalized pressure/force intensity across RGB; Qwen's native temporal
  convolution learns changes between frames instead of receiving a handcrafted
  delta color plane.
- Nearest-neighbor expand each 16×16 tactile map to one 32×32 merger cell.
- Place the 13 aligned right-force summaries in an adjacent 32×32 cell.
- Qwen's temporal kernel fuses adjacent frame pairs.
- Preserve signed v2 calibration values; do not clamp negatives.

An `F`-frame clip produces `2 × ceil(F/2)` output tokens: two spatial tokens
(tactile and force) per temporal frame pair, with an odd final frame repeated as
Qwen requires. Production therefore uses at most 256 tactile tokens per
recording; the smoke test deliberately retains the smaller 64-frame cap.

Set `model.ts_representation=image` for the all-image ablation or `video` for
the all-video ablation.
