# SFT data pipeline (answer-blind zero-shot, 50:50 SFT/RL split)

The split (`split_sft_rl.py`; ratio from `sft_frac` in config.json, group-level,
stratified by (data_source, label)) puts half of each family in SFT and half in RL. Every SFT
row gets a teacher trace: the teacher sees the family context
(the versioned prompt block in `gen_sft_targets.py`), the original question (options and all), and the query
media — never the answer or demonstrations. Up to 4 attempts; the first
completion that passes validation (structure, leak phrases, rationale length,
answer == ground truth under the same `mirl_ext.rewards.combined` scorer RL
uses; ECG additionally requires the verbatim category) is kept, so yield IS
teacher accuracy. One status record per task (accepted/exhausted/error) makes
pass@1/pass@k and per-class yield derivable from the trace file
(`report_traces.py`).

Rows the blind pass exhausts fall back to one `--answer-conditioned` pass over
ALL families (answer revealed, derivation written, marked
`mode=answer_conditioned` in extra_info), so every gradable row ends up traced.
The mode field keeps the two tiers separable downstream: filter to
`answer_blind_zero_shot` when only verified-perception traces should train.

Sources in `schema.OPEN_SOURCES` (haptic_ts descriptions, tactile
captions/notes, free-text video QA) have no exact-match gate and are skipped.

Cluster parquet commands run inside srun (`srun -p cpu -c 8 --mem=32G …`);
`data/sft/` is not synced — scp task/trace files.

```bash
# cluster: split (ratio = sft_frac in config.json), export every SFT row, stage media.
# Use a NEW staging root: cache/source/frame-recipe mismatches fail closed.
python mirl_ext/sft/scripts/split_sft_rl.py --out-root $DATA/split_grpo
# Derive closed time-series MCQs after the recording-locked split.
python mirl_ext/sft/scripts/export_sft_tasks.py \
    --out $DATA/split/sft_tasks.jsonl \
    --families climb_train ecg_train human_behaviour_train tactile_train
python mirl_ext/sft/scripts/stage_media.py --tasks $DATA/split/sft_tasks.jsonl \
    --out-root $MIRL_SCRATCH_ROOT/data/sft_media_v4 --workers 16 --max-side 1536
# rsync sft_tasks.staged.jsonl + media to the laptop

# Tactile teacher traces use the complete synchronized RGB+heatmap composite,
# not separately rendered `.pt` collages: ~1 fps, min 4, max 24 frames.
python mirl_ext/sft/scripts/stage_tactile_v2.py \
    --tasks $DATA/split/sft_tasks.jsonl \
    --out-root $MIRL_SCRATCH_ROOT/data/sft_tactile_v10_mmtouch_min4/media
# emits $DATA/split/sft_tasks.tactile_v2.jsonl

# internet-capable cluster CPU node: preview (no API call), generate (billed;
# rerun = resume), report
python mirl_ext/sft/scripts/gen_sft_targets.py --tasks data/sft/sft_tasks.staged.jsonl \
    --out data/sft/traces_v4.jsonl --image-root data/sft/media_v4 --dry-run
python mirl_ext/sft/scripts/gen_sft_targets.py --tasks data/sft/sft_tasks.staged.jsonl \
    --out data/sft/traces_v4.jsonl --image-root data/sft/media_v4
# Coverage pass: accepted blind traces stay untouched; exhausted/error rows retry
# with the verified answer exposed and are marked mode=answer_conditioned.
python mirl_ext/sft/scripts/gen_sft_targets.py --tasks data/sft/sft_tasks.staged.jsonl \
    --out data/sft/traces_v4.jsonl --image-root data/sft/media_v4 --answer-conditioned
python mirl_ext/sft/scripts/report_traces.py data/sft/traces_v4.jsonl --audit 8

# cluster: after copying the trace back, freeze exactly the teacher's images/frames,
# join only unchanged source rows, then audit every parquet + training-media byte.
python mirl_ext/sft/scripts/select_trace_frames.py \
    --tasks $DATA/split/sft_tasks.staged.jsonl \
    --traces $DATA/split/traces_v4.jsonl \
    --out-root $MIRL_SCRATCH_ROOT/data/sft_frames_v4
python mirl_ext/sft/scripts/build_sft_parquet.py \
    --traces $DATA/split/traces_v4.jsonl \
    --media-from-staging $MIRL_SCRATCH_ROOT/data/sft_frames_v4
python mirl_ext/sft/scripts/audit_sft_parquet.py \
    --parquet-root $DATA/split_grpo/sft_parquet \
    --media-root $MIRL_SCRATCH_ROOT/data/sft_frames_v4 \
    --traces $DATA/split/traces_v4.jsonl
SMOKE=1 sbatch mirl_ext/sft/run_sft_b200.sbatch
```

The builder keeps every accepted row from the SFT half in training. It does
not create another internal validation split; evaluation uses the untouched
task validation files outside `split_grpo/sft/`.

Do not reuse legacy staged tasks, trace files, frozen-frame roots, or audit
manifests after changing a split, prompt/media row, frame count, image size, or
source file. Source-row fingerprints and SHA-256 manifests intentionally stop
instead of silently joining or training stale artifacts. The launcher accepts
only the four explicit closed-task parquets and refuses a Hydra override of
`data.train_files`.
The launcher appends the audit SHA to `EXPERIMENT_NAME`, so changed data cannot
resume an old dataloader/checkpoint. Set a new `EXPERIMENT_NAME` prefix (or
`RESUME_MODE=disable`) when changing the base model or training hyperparameters.

Tests: `python mirl_ext/sft/tests/test_sft_pipeline.py`.

Before real training: set `data.max_length` (yaml default 1024 is far too
small); Qwen3.5's template rejects a lone system turn, so `build_sft_parquet.
sft_messages()` merges it into the user turn (known train/serve difference vs
GRPO).

Native signals remain unintegrated: `split_grpo` time-series rows are rendered
plots (`signals` dropped at rewrite; refs recoverable via
`ts_images/<ds>_map.json`). Build the aligned-Qwen checkpoint with
`mirl_ext/alignment/export_stage1_vision.py`; a signals → pseudo-video SFT mapping is still
missing, so ts-family SFT trains the plot baseline, same as GRPO.
