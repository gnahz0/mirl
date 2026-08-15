# SFT data pipeline (v1: answer-blind zero-shot)

The teacher gets the family context (`teacher_context.py`), the task's label
definitions, the original question, and the query media — never the answer or
demonstrations. Up to 4 attempts; the first completion that passes validation
(structure, leak phrases, rationale length, answer == ground truth under the
same `mirl_ext.rewards.combined` scorer RL uses) is kept, so yield IS teacher
accuracy. One status record per task (accepted/exhausted/error) makes pass@1,
pass@k, and per-class yield derivable from the trace file (`report_traces.py`).
Generation reads `split_grpo/sft/` only.

Pipeline: `export_sft_tasks.py` → `stage_media.py` → `gen_sft_targets.py` →
`report_traces.py` → `build_sft_parquet.py`. The 50:50 split (`split_sft_rl.py`)
already ran; don't re-run it. `gen_sft_episodes.py` (few-shot smellnet
episodes, 98% yield) is not part of v1 — the fallback if zero-shot smellnet
comes back poor.

What the data dictates:
- smellnet exports only the 50-class single-substance task (121 SFT-half rows);
  mixtures/GC-MS are excluded and asserted absent.
- Open-response sources (all of haptic_ts, tactile notes/captions, free-text
  human-behaviour QA) can't be exact-match validated → exported as
  `answer_style: open` and skipped.
- Time-series rows in `split_grpo` ARE rendered plots; the native-signal
  student path is not built yet (needs: alignment-tower → full-HF-checkpoint
  exporter, and a signals → pseudo-video dataset mapping; recipe exists in
  `alignment/model.py::_timeseries_frames`). Until then ts-family SFT trains
  the plot baseline, same as GRPO.

Cluster parquet commands run inside srun (`srun -p cpu -c 4 --mem=32G …`);
`data/sft/` is not synced — scp task/trace files.

```bash
# cluster: export a balanced pilot (~200 rows/family, water-filled) + stage media
python mirl_ext/sft/export_sft_tasks.py --limit-per-family 200 --out $DATA/split/pilot_tasks.jsonl
python mirl_ext/sft/stage_media.py --tasks $DATA/split/pilot_tasks.jsonl --out-root $DATA/split/media
# scp pilot_tasks.staged.jsonl + media/ to the laptop

# laptop: sanitized request preview (no API call), then generation (billed!), then report
python mirl_ext/sft/gen_sft_targets.py --tasks pilot_tasks.staged.jsonl \
    --out data/sft/pilot_traces.jsonl --image-root data/sft/media --dry-run
python mirl_ext/sft/gen_sft_targets.py --tasks pilot_tasks.staged.jsonl \
    --out data/sft/pilot_traces.jsonl --image-root data/sft/media   # rerun = resume
python mirl_ext/sft/report_traces.py data/sft/pilot_traces.jsonl --audit 8

# cluster: build parquets (asserts join gt + placeholder/media counts), then a
# tiny trainer run as the smoke test before any long launch
python mirl_ext/sft/build_sft_parquet.py --traces $DATA/split/pilot_traces.jsonl
torchrun --nproc_per_node=1 -m verl.trainer.sft_trainer model.path=$QWEN35 \
    data.train_files=$DATA/split_grpo/sft_parquet/climb_train_sft.parquet \
    data.max_length=16384 data.train_max_samples=16 trainer.total_epochs=1
```

Full generation = the same steps with `--limit-per-family 2500`.
Tests: `python mirl_ext/sft/tests/test_sft_pipeline.py`.

Before real training: set `data.max_length` (yaml default 1024 is far too
small); Qwen3.5's template rejects a lone system turn, so `build_sft_parquet.
sft_messages()` merges it into the user turn (known train/serve difference vs
GRPO); climb mixes image and video rows in one parquet — if verl's collator
chokes on the mix, split them into two files.
