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
`gen_sft_episodes.py` (few-shot smellnet episodes, 98% yield) is the fallback —
zero-shot smellnet measured 0% even with the substance descriptions.

Cluster parquet commands run inside srun (`srun -p cpu -c 8 --mem=32G …`);
`data/sft/` is not synced — scp task/trace files.

```bash
# cluster: split (ratio = sft_frac in config.json), export every SFT row, stage media
python mirl_ext/sft/scripts/split_sft_rl.py --out-root $DATA/split_grpo
python mirl_ext/sft/scripts/export_sft_tasks.py --out $DATA/split/sft_tasks.jsonl
python mirl_ext/sft/scripts/stage_media.py --tasks $DATA/split/sft_tasks.jsonl \
    --out-root $MIRL_SCRATCH_ROOT/data/sft_media --workers 16 --max-side 1536
# rsync sft_tasks.staged.jsonl + media to the laptop

# laptop: preview (no API call), generate (billed; rerun = resume), report
python mirl_ext/sft/scripts/gen_sft_targets.py --tasks data/sft/sft_tasks.staged.jsonl \
    --out data/sft/traces.jsonl --image-root data/sft/media --dry-run
python mirl_ext/sft/scripts/gen_sft_targets.py --tasks data/sft/sft_tasks.staged.jsonl \
    --out data/sft/traces.jsonl --image-root data/sft/media
python mirl_ext/sft/scripts/report_traces.py data/sft/traces.jsonl --audit 8

# cluster: build parquets, then a tiny trainer run as the smoke test
python mirl_ext/sft/scripts/build_sft_parquet.py --traces $DATA/split/traces.jsonl
torchrun --nproc_per_node=1 -m verl.trainer.sft_trainer model.path=$QWEN35 \
    data.train_files=$DATA/split_grpo/sft_parquet/climb_train_sft.parquet \
    data.max_length=16384 data.train_max_samples=16 trainer.total_epochs=1
```

The builder keeps every accepted row from the SFT half in training. It does
not create another internal validation split; evaluation uses the untouched
task validation files outside `split_grpo/sft/`.

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
