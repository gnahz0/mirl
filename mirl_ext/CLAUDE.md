# Working in `mirl_ext/`

Agent-facing operating rules for the MIRL extension package. Cluster paths,
accounts, SSH, and submit commands live in the untracked root `CLAUDE.local.md`
and `mirl.env` — nothing identifying belongs in tracked files. `source
mirl.env` before running or submitting anything cluster-side.

## Fork boundary — read first

This repo is a **fork of verl**. `AGENTS.md` and the root `CLAUDE.md` symlink
that points at it are upstream files: do not edit them, do not break the
symlink. MIRL behavior belongs in `mirl_ext/`, never as a patch to verl
internals. `rewards/_common.extract_boxed_answer` delegates (lazily) to
`verl.utils.reward_score.math_dapo`: an upstream rebase touching that file
silently shifts RL reward and the SFT keep-first-correct gate — re-verify on
rebase. The one sanctioned exception is documented in
`docs/mirl/README.md` (`agent_loop.py`, Qwen3.5 video token expansion) —
extend that list only with a matching note there.

## Stage-1 alignment invariants

`alignment/` trains the Qwen3.5 vision tower on sensor pseudo-videos while
preserving image understanding; rendering lives in `mirl_ext/data/signals.py`
(shared with the future native-signal SFT/RL path). Things that look like
style but are load-bearing:

- **Loss reductions.** Distillation uses cosine similarity with
  `torch.segment_reduce` so every visual sample weighs equally despite
  different token counts. ECG sums candidate-pair losses per anchor, then
  takes a class-balanced anchor mean. Each tactile task uses the
  global mean over its observed sample-label pairs, and the six task losses
  average equally. Changing a reduction silently rebalances training.
- **SigLIP uses complete fixed label banks**: ECG 7 and six closed tactile
  QA banks of 6/6/4/2/8/4 choices (initial-contact and
  highest-pressure may have multiple positives). `log_logit_scale` and
  `logit_bias` init to log(10) and -10 and train **without weight decay**
  (the ndim<=1 optimizer group in `train.py` enforces this).
- **Rendering**: one uniform `normalize()` for every family — z-score per
  channel over time, no per-family branches and no "already normalized"
  detection (z-scoring is idempotent, so pre-z-scored ECG comes out unchanged
  as a mathematical consequence, not a special case); tactile is z-scored
  over the whole recording (per-taxel scaling would amplify untouched-taxel
  noise). Both clamp at 4
  sigma into [-1, 1]; constant rows map to zero; non-finite input raises.
  Semantic boundaries land on 32 px (patch 16 x merger 2x2) so a merged token
  never spans two channels; -1.0 is the tail-padding value. Qwen's own video
  processor runs with resize/rescale/normalize disabled.
- **Batches are source-homogeneous** (one media kind, one `data_source`).
  Low-resource signal sources repeat complete shuffled passes via integer
  `train.signal_repeat_factors` (approximately square-root sampling);
  validation is one-pass; the sampler skips groups too small to give every
  rank a sample. Image and RGB-video rows are preservation anchors — their
  annotation text is ignored for that loss. Human-behavior and CLIMB videos
  use at most 8 frames; tactile composites use approximately 1 fps with a
  4-frame floor and 24-frame ceiling. Native sensor sequences stay dense.
- **Metrics carry no placeholders.** A key is present iff its branch fired;
  never pre-populate `loss/*` with 0.0. The cross-rank reduction uses the
  static `_REDUCED_METRIC_KEYS` list — deriving keys from a step's dict
  deadlocks when ranks disagree. Selection uses `val-core/map/overall`
  (equal-family mean over ECG and tactile).
- **Checkpoint config keys are exact file paths, never directories**:
  `train.init_checkpoint` -> an `alignment_state.pt` (weights-only, fresh
  schedule); `train.resume_checkpoint` -> the matching `last/trainer_state.pt`
  (same sampler geometry and planned schedule).

**When changing the objective or rendering: start a new lineage.** Bump
`WANDB_RUN_ID` and the checkpoint dir; losses and val metrics are not
comparable across lineages.

## Running it

```bash
# single GPU through the same torchrun path
NUM_GPUS=1 bash examples/alignment/run_stage1_qwen35_siglip2.sh
# multi-GPU
torchrun --standalone --nproc_per_node=4 -m mirl_ext.alignment.train --config <same>
# cluster
source mirl.env && sbatch --output "$MIRL_CLUSTER_ROOT/mirl-qwen35/logs/stage1_%j.out" \
  mirl_ext/alignment/run_stage1_b200.sbatch
```

Config keys are OmegaConf-overridable on the CLI. Warmup is a fraction of the
run (`warmup_ratio`); `val_every` is an optimizer-step interval. Validation
saves `best/` and a resumable `last/`; the run also saves one final encoder.

## Measured lessons (do not re-derive; re-measure if the regime changes)

- **Measurement beats derivation.** Every major mistake here came from
  reasoning over an available measurement (a cited lr vs our own logs; "the
  val mix is harder" vs an eval showing the opposite; an upscale a probe
  disproved). Probe first; it costs minutes.
- **Dimensional collapse is the historical failure mode**: training traded
  embedding rank for margin on every family while every logged metric looked
  healthy. Effective-dimension probes are diagnostics, not training metrics.
- **ECG is not a working positive control**: the majority class is ~44% of the
  filtered val set — always quote margins over the MAJORITY baseline — and
  label prevalence is strongly corpus-dependent.
- **Per-family evals are mandatory.** Mixed validation once hid chance-level
  tactile behavior.
- **Only the encoder transfers to Stage 2.** `distill` is the learn-vs-
  preserve knob (image cosine above 0.99 guardrail).
- `num_workers` x prefetch is a host-RAM multiplier for video decode, not
  throughput.
- Never inspect parquet on the login node — wrap in `srun` (details in
  `CLAUDE.local.md`); the OOM kill is silent over non-interactive ssh.
- **A rank desync at a "deterministic step" is resource exhaustion until
  proven otherwise.** The p20 step-271 saga (6 dead runs, 5 disproven
  distributed-systems theories: NCCL flags, modality-asymmetric collectives,
  kernel hangs, step-indexed branches, batch content) was two resource walls:
  (a) the SFT dataloader host-RAM ratchet (~0.12 GB/batch/rank, measured)
  crossing the job's 128G `--mem` cap at a data-position-determined batch —
  an OOM that presents as a deterministic landmine; (b) the shared-account
  /scratch QUOTA blowing (blocked writes desync ranks the same way; also
  kills jobs at 0s — no log — and stalls trace generation). Identical config
  with `--mem=384G` + healthy quota trained 846/846 clean. Symptom key: one
  rank EOFErrors in the metrics all_gather, the other wedges; MaxRSS pinned
  at the cap in sacct is the tell. Check `sacct -o MaxRSS` and `lfs quota`
  BEFORE any collective-level theory. (Honest confound: the completed run
  also used `DistributedModalityHomogeneousSampler` via
  mirl_ext/sft/sft_trainer.py — never isolated from the resource fixes; it
  stays as cheap defense-in-depth.)

## Upstream recipes — check BEFORE inventing a representation

Every family has a published recipe that contradicts some choice this code
makes; read the relevant one before changing a data path.

- **Tactile is TRUNCATED to 24 s corpus-wide** (decision 2026-09-02): the
  >24 s tail (~94 recordings, 4%) is repeated squeeze cycles carrying no new
  signal. Truncated twins live beside the originals with IDENTICAL basenames
  (`truncated24/` mp4s, `truncated24_pt/` taxel+force tensors); every tactile
  train/valid parquet AND the alignment `haptic_ts_*` parquets reference the
  twins (backups: `*.pre_trunc24.parquet`; full originals untouched).
  For teacher traces, `stage_tactile_v2.py` keeps each video's full synchronized
  RGB+heatmap composite and samples approximately 1 fps over the same window.
  Clips under 4 s get four distinct uniformly spaced frames; all clips include
  their first and final frame. Source `.pt` tensors remain the dense tactile
  stream used by alignment and are not teacher-media staging artifacts.
- **Tactile** ([OpenTouch](https://opentouch-tactile.github.io/),
  [arXiv:2512.16842](https://arxiv.org/abs/2512.16842)): upstream has **no
  text modality** — semantics come from a closed 29-class grasp taxonomy; the
  captions are annotation metadata. They explicitly ablated
  big-encoder-on-upscaled-maps and it lost to a small from-scratch CNN. Their
  scale is global and fixed; captions were generated from peak-pressure
  frames.
- **ECG** (data via [DDVD233/CLIMB](https://github.com/DDVD233/CLIMB), which
  vendors ECG-JEPA): CLIMB does not use a vision encoder for ECG, and its
  dataset classes normalize only by the recording's global peak-to-peak range
  (no filter, no z-score — those exist only in its MIMIC pipelines). Standard
  practice is per-lead z-score, Butterworth 1-47 Hz, 2-3 s crops with TTA.
  ECG diagnosis is inherently multi-label, but the delivered
  `ecg_train.json` already carries exactly ONE superclass per record
  (verified: zero multi-label rows) — the flattening happened in CLIMB's
  preprocessing, not in this repo; our parquets are faithful to it. So
  single-label top-1 is mis-specified by inheritance — always quote the
  0.441 majority baseline; macro AUROC would be proper; single-run f1
  deltas under ~0.01 are noise.

## GRPO data protocol (launcher is run_qwen35_grpo.sh — trust it, verify lists)

- The old extra_info-JSON-string blocker is RESOLVED: `MIRLDataset`
  (mirl_ext/data/dataset.py, wired via data.custom_cls) decodes it plus does
  the load-bearing media capping — never add a second RL dataset class.
- **RL trains ONLY on the RL half** (`split_grpo/rl/`, via `join_train`/
  `TRAIN_ROOT`); the root-level `<fam>_train.parquet` files are the UNSPLIT
  corpora and overlap the SFT half (bug fixed 2026-08-29 — it had trained on
  full corpora). Validation files live at the root, never split.
- **RL touches gradable sources only**: tactile/human_behaviour use the
  `_closed` variants (open free-text stripped, was 28% of each); haptic_ts
  is excluded entirely (100% `haptic_tactile` open captions). haptic_mcq is
  RETIRED (2026-09-01): synthetic plot-MCQ rows are not part of SFT or RL;
  `haptic_mcq_valid.parquet` remains on disk only as a ts-plot eval artifact.

## Gotcha: `prompt` is a MESSAGE LIST, not a string

climb/tactile carry two messages (system + user); ecg/haptic_ts carry
one. The media placeholder and the question live in the USER turn, so reading
`prompt[0]` silently drops both — use `mirl_ext.data.schema.prompt_messages()`.
Qwen3.5's chat template rejects a system-only list, so
`build_sft_parquet.sft_messages()` merges system into the user turn (a known
train/serve difference vs GRPO, documented there).

## SFT v1 protocol (50:50 SFT/RL split)

`mirl_ext/sft/README.md` is authoritative. The split unit is a GROUP (the
underlying recording), never a row; ratio = `sft_frac` in `sft/config.json`.
Teacher generation is answer-blind zero-shot first (up to 4 attempts,
keep-first-correct under the same scorer RL uses, so blind yield IS accuracy);
rows that exhaust fall back to one answer-conditioned pass over all families,
marked `mode=answer_conditioned` so the tiers stay separable. Open sources are
skipped (no exact-match gate). **Task uids are positions within
the split-half parquet — any re-split invalidates every old uid; never resume
an old trace file across splits.** The native-signal student path is not yet
integrated: ts-family SFT currently trains on rendered plots.

**Open-response rows are EXCLUDED from SFT training** (decision 2026-08-22,
pending caption-provenance verification): drop every `*_open_sft.parquet`
from the sbatch file lists. The p20-dyn pilot predates this and has them baked
in. `--open-gt` still builds the parquets; exclusion happens in TRAIN_FILES.
SFT has no additional internal validation split; use the untouched task
validation files for evaluation.
