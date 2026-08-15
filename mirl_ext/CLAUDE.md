# Working in `mirl_ext/`

Agent-facing operating rules for the MIRL extension package. For *what this is*
and the job/environment history, read [`docs/mirl/README.md`](../docs/mirl/README.md)
and [`docs/mirl/CONTINUATION.md`](../docs/mirl/CONTINUATION.md) — this file
deliberately does not repeat them. Private cluster paths, SSH, and the shared
account rules live in the repo-root `CLAUDE.local.md` (gitignored).

## Fork boundary — read first

This repo is a **fork of verl**. `AGENTS.md` and the root `CLAUDE.md` symlink
that points at it are **upstream files** (verl PR #5846). Do not edit them and
do not break the symlink: upstream is merged in periodically, so any local
change there becomes a recurring conflict.

MIRL behavior belongs in `mirl_ext/`, never as a patch to verl internals. The
one sanctioned exception is documented in `docs/mirl/README.md`
(`agent_loop.py`, Qwen3.5 video token expansion) — extend that list only with a
matching note there.

## Stage-1 alignment invariants

`alignment/` trains the Qwen3.5 vision tower on time-series pseudo-videos while
preserving image understanding. These are the things that look like style but
are load-bearing:

**Loss reductions are load-bearing.** Distillation uses
`F.cosine_similarity` and `torch.segment_reduce` so every visual sample has equal
weight despite different token counts. SmellNet and ECG sum candidate-pair losses
per anchor before taking a class-balanced anchor mean. Each tactile task uses the
global mean over its observed sample-label pairs, and the six task losses are
averaged equally. Changing a reduction silently rebalances training.

**SigLIP uses complete fixed text-label banks.** SmellNet has 50 labels and ECG
has 7. Tactile has six closed QA banks with 6, 6, 4, 2, 8, and 4 verbalized
choices; initial-contact and highest-pressure targets may have multiple positives.
The canonical shared `log_logit_scale` and `logit_bias` parameters initialize to
`log(10)` and `-10`, respectively, and both train without weight decay.

**The clean baseline is sensor-to-text only.** SmellNet and ECG use their stored
class labels. Tactile joins the six closed QA answers to each pressure recording
by recording stem and aligns against concise verbalized choices; open answers
remain available for SFT but are not Stage-1 targets. SmellNet mixtures, GC-MS,
and tactile-video alignment do not enter the objective.

**Visual rows are preservation anchors, not QA examples.** Their annotation text
is ignored, so `AlignmentDataset` expands multi-media rows and keeps one row per
unique image/video path. Visual rows remain one-pass; low-resource signal sources
may repeat complete shuffled passes through integer
`train.signal_repeat_factors`. Validation remains one-pass. The sampler skips
source groups too small to give every rank a sample. Every global microbatch has
one media kind and one `data_source`. Images that PIL identifies specifically as
truncated get one scoped permissive-decode retry. Remaining recognized media-load
errors are logged and filtered per item; W&B reports per-kind, total, cumulative,
and fractional skip statistics. An entirely unreadable rank-local batch still
fails rather than passing an empty batch into DDP. Programming and schema errors
remain visible.

The current repeat factors approximate square-root sampling rather than equal
family exposure. Unified rendering does not require equal family frequency;
making 250-row SmellNet exactly match 39,445-row ECG would heavily repeat the
small set. Tactile's six task losses are still equally weighted within each
tactile batch.

**Do not reuse Stage-1 media flattening for SFT or RL.** Those stages consume the
annotation and must retain each dataset row as one example, load every image or
video in its original order, and keep the media count aligned with the prompt
placeholders. In the current CLIMB train data, multi-image rows contain either two
or four images with the same number of `<image>` placeholders; using only
`images[0]` would silently discard supervision.

**Labels are balanced explicitly.** SmellNet and ECG anchors with the same label
share that label's total row weight. Tactile averages each task over globally
observed rows, then averages the six tasks. Counts are global within each
source-homogeneous microbatch.

**Selection metrics use one uniform family surface.** W&B publishes accuracy,
Recall@1, Recall@5, and mAP for SmellNet, ECG, and tactile. Accuracy means the
top-ranked candidate is positive. Recall@k is the
fraction of ground-truth positives recovered in the top k, so it equals accuracy
at k=1 for single-label tasks but can be lower for multi-positive tactile tasks.
`overall` is the equal-family mean, and overall mAP selects the best checkpoint.
Training metrics cover the current accumulation window; validation uses the full
set.

**Metrics carry no placeholder values.** A key is present iff its branch fired.
Never pre-populate `loss/*` with `0.0`: a placeholder is indistinguishable from
a measurement and corrupts both the cross-rank average and the val means. The
cross-rank loss reduction uses the **static** `_REDUCED_METRIC_KEYS` list —
deriving keys from a step's dict deadlocks when ranks disagree. Losses are
averaged over ranks that computed them; `n/*` counts are summed.

**Pseudo-videos use Qwen's own video processor.** Resize, rescale, and image
normalization stay disabled because sensor tiles are already normalized and
aligned. Semantic boundaries must land on 32px (patch 16 × merger 2×2) so a
merged token never spans two sensors/leads.

**Zero MAD falls back to the full standard deviation.** SmellNet quantization
and sparse tactile traces can have `MAD=0` while `std>0`. In that case robust
normalization uses `std`, not the usual `0.7 * MAD + 0.3 * std` blend; otherwise
a one-count excursion is artificially amplified by `1 / 0.3`. Truly constant
rows still map to zero.

## Running it

```bash
# single GPU through the same torchrun path
NUM_GPUS=1 bash examples/alignment/run_stage1_qwen35_siglip2.sh
# multi-GPU
torchrun --standalone --nproc_per_node=4 -m mirl_ext.alignment.trainer --config <same>
# cluster
sbatch examples/mirl/slurm/run_stage1_b200.sbatch
```

Config keys are OmegaConf-overridable on the CLI (`train.num_train_epochs=2`).
The trainer derives optimizer steps per epoch as
`ceil(len(train_loader) / grad_accum_steps)` and flushes the final partial
accumulation window. `val_every` is an optimizer-step interval; warmup is
configured as a fraction of the complete run with `warmup_ratio`. Validation
saves the best encoder and a resumable `last/` checkpoint containing optimizer
and scheduler state; the run also saves one lightweight final encoder.
`train.init_checkpoint` is a weights-only continuation with a fresh schedule.
`train.resume_checkpoint` requires the same sampler geometry and planned total
schedule as the saved `last/` state.

## When changing the objective

Start a new lineage: bump `WANDB_RUN_ID` *and* the checkpoint dir in the sbatch.
`loss/*` is not comparable across objectives; prediction metrics read cosine
similarities rather than the loss. Aggregate and per-family losses live with selection metrics under
`val-core/`; component losses, coverage, and per-class diagnostics
live under `val-aux/`.
`val-core/{accuracy,f1_macro,recall_at_1,recall_at_5,map}/overall` is the
equal-family mean across SmellNet, ECG, and tactile. Macro-F1 excludes classes
absent from that validation sample.

## Experiment log (2026-07-27..30) — what was tried, with verdicts

Runs (val/f1 on the protocol noted; "fixed" = original val mix, n_ts=519):

| run | change | f1 (fixed) | verdict |
|---|---|---|---|
| v1/v2 | InfoNCE, lr 1e-5, clip 1/5 | 0.478 / 0.446 | best f1; InfoNCE optimizes the argmax metric directly |
| v4 | sigmoid, logit_bias=-10 | 0.135 | COLLAPSED in that setup; canonical learned-bias training is a new lineage |
| v7 | sigmoid calibrated, lr 3e-5, gather | 0.301 | lr 3e-5 hurt (~2x f1 deficit vs 1e-5, measured) |
| v8 | lr 1e-5, clip 1, oversample 3 | 0.309 | completed; encoder drift only 3.3e-3 (heads did the work) |
| v9 | continue + distill_img 0.5 | 0.388/0.587 (own mix, peak) | encoder freed (drift +22% in 300 steps); killed at 600 for v11 |
| v11 | + scalar_lr, haptic fixes, distill 0.25 | 0.335 @ step 300 | FLAT for 300 steps below v9's peak; diagnosed 2026-07-30, see below |

## 2026-07-30 diagnosis — the plateau is DIMENSIONAL COLLAPSE

Historical measurements on the v9/best checkpoint and pristine tower used 96
validation recordings per family:

| family | classes | eff-dim VE raw | eff-dim projected | offdiag cos | top-1 | baseline |
|---|---|---|---|---|---|---|
| ecg pristine | 7 | 13.2 | 13.2 | 0.964 | 0.063 | 0.441 (majority) |
| **ecg trained** | 7 | 2.7 | **1.7** | 0.721 | 0.500 | 0.441 (majority) |
| smellnet pristine | 56 | 10.3 | 10.6 | 0.919 | 0.031 | 0.018 |
| **smellnet trained** | 56 | 1.8 | **1.2** | 0.801 | 0.021 | 0.018 |
| haptic pristine | 96 | 8.2 | 7.5 | 0.964 | 0.010 | 0.010 |
| **haptic trained** | 96 | 1.3 | **1.0** | 0.833 | 0.021 | 0.010 |

**Training trades RANK for MARGIN on every family.** Effective dimensionality
collapses (10.3 -> 1.8, 13.2 -> 2.7, 8.2 -> 1.3) while the within-vs-between
margin improves, so every previously-logged metric looked healthy. In the
projected space the loss optimizes, everything lands on ~1 axis. A ~2-D embedding
hosts 7 coarse ECG classes; it cannot host 93 smellnet or 635 haptic labels.
**No reweighting of the loss recovers axes the projection has already discarded.**
The clean baseline therefore uses Qwen's 1152-D pre-merger encoder state directly and the two essential
losses before testing anti-collapse methods separately. The effective-dimension
measurements above remain historical diagnostics, not training-loop metrics.

**ECG is NOT a working positive control.** "Normal" is 44.1% of
`ecg_valid_nan_filtered`, so 0.500 top-1 is ~6 points over a constant-majority
predictor, not 3.5x chance, and the probe predicted only 4 of 7 classes. Label
prevalence is also strongly corpus-dependent (MI: ptbxl 1314 / Georgia 366 /
Chapman 10 / CPSC 12; AF: Georgia 14 vs Chapman 1021), so part of any margin is
predicting the source corpus's prior. Always quote the margin over the MAJORITY
baseline — the probe prints both.

**Historical tactile caption baseline (replaced 2026-08-08).** The old target was
one unique open answer per recording (1575/1575 train, 635/635 valid), making
validation exact-caption retrieval. Stage 1 now uses the six reusable closed QA
tasks above; open answers are reserved for SFT.

**SmellNet's old raster destroyed its most informative channels.** Measured
37.3% padding, 38.7% saturated, only 4.3 post-merger tokens/recording. Mechanism
(verified on raw CSVs, and NOT "dead sensors"): the base sensors are integer-valued
with a 1-10 count dynamic range, so >50% of samples sit at exactly the median,
`MAD` is **exactly 0**, `scale` collapses to `0.3*std` ~ 0.05-0.2, and a +-1-count
quantization step saturates to +-1. A truly constant channel renders 0 (harmless);
the killer is near-constant. Worst observed: `avocado_6` C2H5OH 47.6% saturated.
Upstream's masking ablation ranks exactly these channels most important
(LPG -28.9, Alcohol -26.5 with differencing on).
The current temporal renderer keeps every native timestep and falls back to the
full `std` whenever MAD is zero; the measurements above are historical.

**ECG uses temporal tokens.** Every scalar-family token covers about 64 ordered
timesteps from one channel. This replaced the coarse 1,024-step image raster and
keeps ECG, smell, and tactile on Qwen's native temporal patch path.

Measured lessons (do not re-derive; re-measure if the regime changes):

- **Measurement beats derivation.** Every major mistake here came from reasoning
  over an available measurement: lr 3e-5 (SigLIP2 citation vs v1/v2's own logs),
  clip=5 (dismissed v1's clip=1), "val-mix is harder" (eval showed the opposite),
  tactile 2x upscale (probe: worse). Probe/eval first; it costs minutes.
- **Similarity gap is not a selection metric** because it moves with common-mode
  shifts; it is no longer published to W&B. Historical `val/f1` numbers above use
  the old mixed, batch-local protocol and are not comparable to supported-class macro-F1.
- **Per-family evals are mandatory.** Mixed validation once hid chance-level smell
  and tactile behavior. Tactile now reports equal-task structured metrics.
- **Only the encoder transfers to Stage 2**; metrics now read its normalized
  pre-merger representation directly without a throwaway head.
  `distill` is the learn-vs-preserve knob, with guardrail loss below ~0.01
  (image cosine above 0.99).
- **Video pipeline history:** process_video kwarg bug meant NO run before v5
  ever decoded video; fixing it surfaced host-RAM OOM (num_workers x prefetch
  buffers decoded video), a historical NCCL watchdog from rank-0-only validation
  (validation is now sharded), and step-time changes. num_workers/prefetch is a RAM
  multiplier, not throughput.
- **Cluster:** exclude b0028,b0024 (no CUDA) and b0012 (NVLink fault — passes
  single-GPU checks, dies under tp>1). Fairshare is shared-account and has no
  age factor; a running allocation is precious. Slurm spools sbatch at submit —
  CLI overrides frozen; YAML/trainer.py read at runtime.
- **Never inspect parquet on the aicr login node.** `pyarrow ... .to_pylist()`
  gets OOM-killed by the 5 GB cgroup and over non-interactive ssh prints
  *nothing* (exit 137, no stderr) — it reads as "the script produced no output".
  Wrap it: `srun -p cpu -c 4 --mem=32G --time=00:15:00 <env>/bin/python x.py`.
  Login-node `python3` has no pyarrow at all; use `envs/alec-mv/bin/python`.

## ⚠️ GRPO is currently BLOCKED by an upstream verl change (found 2026-07-30)

`extra_info` is stored as a JSON **string** in every MIRL parquet -- `ecg_train`,
`ecg_train_tstok`, `smellnet_train`, `climb_train`, `tactile_train`, and
`qwen35_smoke.parquet`. Current `verl/utils/dataset/rl_dataset.py::__getitem__` does:

```python
if "extra_info" not in row_dict or row_dict["extra_info"] is None:
    row_dict["extra_info"] = dict()
index = row_dict.get("extra_info", {}).get("index", 0)      # AttributeError on a str
```

A string is not None, so the guard never fires. Reproduced live by loading
`data/ecg_train.parquet` through `RLHFDataset`; it will die on the first batch,
smoke included. `verl/` is clean in the worktree and `rl_dataset.py` has taken three
upstream PRs (#6595, #6631, #6789) since the last verified GRPO run (job 172783,
2026-07-21 handoff), so this is an upstream regression, not local breakage.

Fix belongs on the MIRL side per the fork boundary -- either rewrite `extra_info` as
a struct column (`rewrite_media_paths.py` is a schema-preserving template) or
normalize it in a `mirl_ext` dataset subclass. Do NOT patch `verl/`.

## Gotcha: `prompt` is a MESSAGE LIST, not a string

`smellnet`/`climb`/`tactile` carry **two** messages (system + user); `ecg`/`haptic_ts`
carry one. The `<image>`/`<video>` placeholder and the actual question live in the
USER turn, so reading `prompt[0]` silently drops both. That produced smellnet SFT
traces answering a question never asked (no 50-substance option list) and broke
veRL's SFT dataset, which asserts the placeholder count matches `len(images)`.
`rl_dataset` reads the full list, so GRPO was unaffected -- only tooling that
shortcuts to `prompt[0]` breaks. Helpers in `mirl_ext/sft/` use `prompt_messages()`.

Related: `MultiTurnSFTDataset` tokenizes each message IN ISOLATION and Qwen3.5's
template rejects a system-only list ("No user query found in messages"), so an
explicit system turn cannot survive the SFT path -- `build_sft_parquet.sft_messages()`
merges it into the user turn. That leaves a real train/serve difference (GRPO renders
a true `system` turn), documented in that function.

## SFT v1 protocol (2026-08-14): answer-blind zero-shot for every family

`mirl_ext/sft/README.md` is authoritative. Teacher = question + label
definitions + fixed family context (`teacher_context.py`, versioned) +
query media; never the answer, never demonstrations. 4 attempts, keep-first-
correct under `rewards.combined` (the RL scorer), one status record per task
(accepted/exhausted/error) so yield IS accuracy (`report_traces.py`).
Open-response sources (haptic_ts, tactile notes, free-text HB QA) are skipped —
no exact-match gate exists. Few-shot episodes (`gen_sft_episodes.py`) are
explicitly NOT v1; revisit only if smellnet pass@4 comes back poor.
⚠️ Native-signal student path is NOT integrated: `split_grpo` ts rows are
rendered plots (`signals` column dropped at rewrite; recoverable via
`ts_images/<ds>_map.json`). Missing pieces are listed in the README — do not
silently treat plot-trained SFT as "native signals".

## Upstream references — check these BEFORE inventing a representation

Every family here has a published upstream recipe, and each one contradicts some
choice this code makes. Read the relevant one before changing a data path.

**SmellNet** — [MIT-MI/SmellNet](https://github.com/MIT-MI/SmellNet),
[arXiv:2506.00239](https://arxiv.org/abs/2506.00239) **v5** (ICLR 2026; v1/v3 are
materially different papers, and v3's headline is 58.5 not 63.3).
- Base is **6 channels @ 1 Hz**, 600 s sessions, 6 days/substance, 50-way,
  **chance 2%**. Mixture is **4 channels @ 10 Hz**, 60 s, and is **12-dim
  proportion REGRESSION, not classification**.
- Stage-1 alignment is currently **base-only**: the alignment loader removes
  `smellnet_mixture` before label-vocabulary construction, sampling, loss
  computation, and metric logging.
  The active split is 250 train recordings and 50 validation recordings.
- **Temporal differencing at lag p=25 (i.e. 25 SECONDS) is the dominant
  hyperparameter**, worth **+18 to +27 points** on every temporal model — more
  than their entire architecture spread. We do none. CNN 29.5->52.7,
  LSTM 28.8->57.9, their transformer 39.9->56.1.
- Normalization is **per-channel z-score with GLOBAL statistics fit on train**
  (`StandardScaler` over all train windows x timesteps), not per-recording.
- They train on **50-100-sample sliding windows, stride w/2** (one 600-sample file
  becomes 11-23 examples), and truncating to 500 steps BEATS the full recording
  (53.2 vs 50.6). Their code also does undocumented per-recording
  `df - df.iloc[0]` baseline subtraction on base data only.
- Best sensor-only **57.9%** (LSTM w=100 p=25); 63.3% only with GC-MS contrastive
  (binned EI mass spectra, a FooDB prior — they measure no GC-MS). Honest error
  bar is **+-6-8 points** (leave-one-day-out 53.0+-6.0) and Day 6 (our test split)
  is a favorable draw.
- Our label space has defects: `almond 50 garlic 50` and `garlic 50 almond 50` are
  two DISTINCT labels for one recipe; mixture names say `clove`/`orange` where the
  base vocab says `cloves`/`mandarin orange` (576 rows); and mixture odorants are
  **extracts, physically different from the base foods** sharing their label text.
- **The mixture task is REGRESSION upstream, classification here — read this before
  trusting any smellnet number.** Verified in `models/load_data.py`: every filename is
  parsed into a 12-element proportion vector over a fixed palette
  (`ALL_INGREDIENTS = [banana, orange, pear, apple, mango, peach, strawberry, clove,
  coriander, garlic, almond, cumin]`, `label_vector = [pct[ing]/100 for ing in ...]`).
  Because it accumulates into a dict keyed by ingredient it is inherently **order- and
  separator-invariant**: `almond_50_garlic_50`, `garlic50_almond50` and
  `Almond50_Garlic50` all collapse to ONE target. The loss is
  `kl + 0.5*eps_l1(eps=0.2) + 0.5*focal_bce(alpha=0.75, gamma=2)`, and the headline
  metric `thr_acc_nonzero` scores a component correct when `|pred-target| < 0.1`,
  averaged over PRESENT components.
  Consequence: upstream's "50.2 Top-1@0.1" is NOT comparable to our exact-string
  accuracy. `mirl_ext/rewards/smellnet.py` uses `acc = (pred_label == gt_label)` with
  **`sim_weight = 0.0`**, so the jaccard it already computes is discarded and
  `almond70_clove30` vs `almond80_clove20` scores exactly 0 -- same as answering
  "banana".
- **The reward fixes below apply to MIXTURE ONLY. Base is fine as-is -- do not
  "fix" it.** `smellnet_base` has exactly **50 labels with ZERO collisions** under
  `_norm_label`; single-substance ID has no meaningful partial credit ("allspice" vs
  "almond" is simply wrong), and enabling `sim_weight` there would hand `brazil_nut`
  a third of a point for guessing `pili_nut` because both contain "nut" -- rewarding
  label-string gaming rather than reading the sensor. Exact match is correct for base.
- Mixture-only fixes: (a) canonicalize to `{ingredient: pct}` like upstream. Measured
  precisely: **4 recipes are split by separator/component-order convention, all inside
  `smellnet_mixture`, 84 rows (9.7% of mixture rows)** --
  `almond_50_garlic_50`/`garlic50_almond50`, and three `bananaN_mangoM` vs
  `banana_N_mango_M` pairs. (b) give tolerance-based partial credit (upstream's +-0.1
  per present component), because an all-zero reward across a GRPO rollout group means
  std 0 and therefore zero advantage.
- **CAUTION, an earlier note in this file got this wrong:** `clove`/`cloves` and
  `orange`/`mandarin_orange` are **NOT duplicate labels and must NOT be merged**.
  `cloves`/`mandarin_orange` are BASE foods; `clove`/`orange` are the MIXTURE palette's
  fragrance/essential-oil extracts, recorded on a different rig (4ch@10Hz vs 6ch@1Hz).
  Per the paper they are physically different substances that happen to share a name.
  Counting them as split recipes is what inflated the defect estimate to "6 recipes /
  114 rows / 11.6%" and before that to "~20%"; the true within-task figure is 4 / 84 /
  9.7%, mixture only.
- Measured 2026-07-30, gpt-5.6-sol reading the rendered plots with the answer
  WITHHELD (`gen_sft_targets.py --blind`): ECG acc 0.249 / macro-F1 0.232 against a
  0.168 majority baseline; smellnet 0.000; haptic token-F1 0.178. 4-shot
  (`--few-shot 4`, demos carry their own plots) moves ECG to macro-F1 0.234 (noise,
  and it only shifts WHICH class is over-predicted), smellnet to 0.025, haptic to
  0.208. Few-shot teaches the label FORMAT, not perception. 73%->66% of smellnet
  answers are "cannot be determined from the plot alone", which is a fair response:
  88% of smellnet rows are `smellnet_mixture`, whose prompt lists NO options and only
  hints the format with one parenthetical example.
- Measured 2026-08-11, few-shot prediction episodes (`gen_sft_episodes.py`: 5-way
  candidates = gt + 2 same-category + 2 random negatives, 2-3 labelled support plots
  per candidate as individual images, answer withheld): gpt-5.6-sol got **119/121
  base queries right** within 4 attempts (92% first-attempt) — vs 0.000 blind 50-way
  and 0.025 with 4 format demos above. Per-class labelled supports carry perception;
  format demos do not. Traces: `data/sft/ts_traces_episodes.jsonl` (all 50 classes
  appear as query and as negative); the stored `attempts` field lets a downstream
  filter drop potential 5-way lucky guesses (6 rows with attempts>=3).
- Task-prompt provenance, ALL families (verified 2026-08-11 against clones of
  MIT-MI/SmellNet + DDVD233/CLIMB + QoQ_Med + OpenTouch-MIT/opentouch, the SmellNet
  HF dataset, the OpenTouch annotations zip, the historical fork's full history, and
  the raw files on aicr). Three provenance classes:
  (1) **smellnet = fork-authored.** System+user prompts minted verbatim in the
  historical fork's `scripts/smellnet_to_combined.py` (commit ece3eb4d); no upstream
  repo/data carries any prompt text. Facts are faithful to upstream: 50-option list
  set-identical to `models/utils.py` (ours alphabetized), base channels match
  `SENSOR_COLS` verbatim, and the mixture prompt's "**C2H5CH**" is NOT a fork typo --
  it is the literal column header of every upstream mixture CSV
  (`timestamp_ms,NO2,C2H5CH,VOC,CO` on HF; base CSVs correctly say C2H5OH), inherited
  by the renderer's `SENSOR_COLS_MIXTURE` and the prompt. `text_description.json`
  (the 50 substance priors in `data/sft/meta/`) also ships upstream:
  HF `DeweiFeng/SmellNet` `text_data/text_description.json`.
  (2) **haptic = upstream data.** The prompts live verbatim in the raw annotation
  files `/scratch/dvdai_mit/alecz/align_raw/haptic_3ts/annotation_verl_*.json`,
  already verl-formatted by raofu's 3DHaptic pipeline (source
  `/orcd/compute/ppliang/001/raofu/3DHaptic/...` per `build_haptic_ts_jsonl.py`);
  the fork only converts them. The PUBLIC opentouch annotations zip has no prompts
  at all -- just metadata columns + GPT-5 scene captions (the `description` field
  our ground truths come from).
  (3) **ecg = pre-baked upstream of the fork.** Arrived as `problem`/`answer` fields
  in `/scratch/ecg/ts_{train,valid}.json` on the source (mib/engaging) cluster; the
  minting script is unreleased, but the 7 options expand CLIMB's
  `CLASSES = ['normal','cd','mi','sttc','other','afib','hyp']` verbatim IN ARRAY
  ORDER using CLIMB's `to_qa()` house template (CLIMB ships no ECG `to_qa()`).
  The shared `<think>`/`\boxed{}` "internal monologue" boilerplate appears verbatim
  in DDVD233/QoQ_Med `examples/format_prompt/medical_format.jinja` (lineage: verl
  geo3k -> QoQ_Med -> fork, "put in" -> "wrapped in").
  ⚠️ The haptic `annotation_verl_*.json` files exist ONLY on aicr `/scratch`
  (30-day access purge) and in raofu's engaging dir -- they are in no git repo.

**Haptic / tactile** — [OpenTouch](https://opentouch-tactile.github.io/),
[arXiv:2512.16842](https://arxiv.org/abs/2512.16842), code
[OpenTouch-MIT/opentouch](https://github.com/OpenTouch-MIT/opentouch).
- **There is NO text modality.** Contrastive alignment is over
  {video, tactile, pose} only; GPT-5 captions are annotation metadata that is
  never encoded. Semantics come from a **closed 29-class Feix grasp taxonomy**.
  Tactile-only grasp accuracy 60%; their tactile<->video retrieval R@1 is 7.15,
  mAP 15.5. Nobody has shown tactile-text alignment works on this data.
- They **explicitly ablated our architectural bet and it lost**: pretrained
  ResNet-18 on 224x224-upscaled pressure maps vs a 3-layer from-scratch CNN+BiGRU
  at native 16x16 — *"our method improves mAP by 10.49% over ResNet-18... Simply
  enlarging the tactile map and using a large vision encoder does not provide an
  advantage."*
- OpenTouch uses a **global fixed scale**: `clip(x, 0, 3072) / 3072`, with
  negatives clipped to 0. The clean alignment baseline deliberately avoids this
  hard-coded ceiling and robustly normalizes each selected right-map taxel over
  the complete recording instead.
- OpenTouch uses **20 frames @ 30 Hz (0.67 s)**, or 21 frames centered on the
  **peak-pressure frame** — which matters because the captions were generated FROM
  the peak-pressure frames. The alignment baseline intentionally does not copy
  this sampling recipe: it feeds each complete native recording as one example.
- AdamW, lr 1e-4, batch 128-256, 300 epochs, cosine + 5% warmup, tau=0.07 clamped
  at ln(100), **no augmentation at all**.

**ECG** — data comes from [DDVD233/CLIMB](https://github.com/DDVD233/CLIMB)
(`src/datasets/ecg`), which vendors
[sehunfromdaegu/ECG_JEPA](https://github.com/sehunfromdaegu/ECG_JEPA) byte-for-byte
([arXiv:2410.08559](https://arxiv.org/abs/2410.08559)). Checkpoints live on **mib**
at `/scratch/high_modality/ts/{ecg_encoder.pth,sota.pth}` — **not present on
aicr**; they need copying.
- ECG-JEPA: 8 leads x 2500 (= 250 Hz x 10 s, our exact shape) -> **50 temporal
  patches of 50 samples per lead**, 12-layer/768-d/16-head transformer, JEPA
  masked-patch prediction against an EMA target with Smooth L1, 100 epochs on
  Shaoxing + CODE-15. CLIMB mean-pools the 768-d features. **CLIMB does not use a
  vision encoder for ECG.**
- Standard practice: 250 Hz is plenty (100/200/500 Hz interchangeable),
  **3rd-order zero-phase Butterworth 1-47 Hz** (we filter nothing), per-lead
  z-score, **random 2-3 s crops with overlapping-crop TTA** — macro AUC *peaks*
  around 2-3 s and declines beyond, so our full 10 s single view is suboptimal.
- A well-tuned `xresnet1d101` reaches **~0.925 macro AUROC** on PTB-XL-all and a
  **vanilla transformer collapses to 0.857**; 87-97M-param foundation models lose
  to a 2.2M supervised S4. ImageNet-pretrained encoders on rendered ECG images tie
  random init at full data and are worse below it; frozen ImageNet features give
  macro F1 < 0.23. Off-the-shelf CLIP/SigLIP on ECG plots: 81.5-83.7 vs ~90.7 for
  signal SSL.
- ECG diagnosis is inherently **multi-label** (~1/4 of PTB-XL records carry >=2
  superclasses; a 0.94-AUROC model can score 0.21 exact-match), so our
  single-label top-1 is a mis-specified metric. Report macro AUROC, and use
  leave-one-corpus-out — random splits reward learning the corpus prior.
- Noise floor: run-to-run std is 0.0016-0.0039 and bootstrap CI half-widths are
  ~+-0.008, so **any delta under ~0.010 from a single run is noise.** Several of
  our cross-run f1 gaps are inside that band.

The current clean baseline uses each full 8x2500 ECG tensor in both training and
validation. It does not apply the 2-3 second crop described above.
