# SFT data pipeline (v1: answer-blind zero-shot)

Teacher traces for the multimodal SFT stage. The teacher sees the family
context, the task's own label definitions, the original question, and the query
media — never the ground truth, few-shot demonstrations, or a gold-derived
candidate set. Up to 4 independent attempts; the first completion that passes
deterministic validation (structure, leak phrases, rationale length, answer ==
ground truth under the same `mirl_ext.rewards.combined` scorer RL uses) is
kept, so yield IS teacher accuracy. Generation reads `split_grpo/sft/` only;
the RL half and validation are untouched.

Stages: `split_sft_rl.py` (already run; do not re-run) → `export_sft_tasks.py`
→ `stage_media.py` → `gen_sft_targets.py` → `report_traces.py` →
`build_sft_parquet.py` → `smoke_sft_load.py`. Fixed teacher prompts live in
`teacher_context.py` (`PROMPT_VERSION` + hash stamped into every record).
`gen_sft_episodes.py` (few-shot episodes) is NOT part of v1 — kept for a later
decision. Cluster paths come from `config.json` / `MIRL_*` env overrides.

Scope notes, measured from the data:
- smellnet exports the 50-class single-substance task only (121 SFT-half rows);
  mixture and GC-MS rows are excluded and asserted absent.
- Open-response sources (all of haptic_ts, tactile captions/notes, free-text
  human-behaviour QA) have no exact-match gate and are skipped by default
  (`answer_style: open` in the export).
- Time-series families (smellnet/ecg/haptic_ts) carry rendered plot PNGs as
  their student media in `split_grpo`; the native-signal student path is not
  integrated yet (see "Native signals" below).

All parquet-touching commands run on the cluster **inside srun** (login node
has a 5 GB cgroup): `srun -p cpu -c 4 --mem=32G --time=00:30:00 python …`.
`data/sft/` is not Mutagen-synced — move task/trace files with scp.

## Commands

```bash
# 1. Schema audit (cluster)
python mirl_ext/sft/audit_schema.py

# 2. Balanced pilot export, ~200 rows/family water-filled over (source, label) (cluster)
python mirl_ext/sft/export_sft_tasks.py --limit-per-family 200 \
    --out $DATA/split/pilot_tasks.jsonl

# 3. Teacher media staging (cluster; frames per video from the row's max_frames)
python mirl_ext/sft/stage_media.py --tasks $DATA/split/pilot_tasks.jsonl \
    --out-root $DATA/split/media
#    then scp pilot_tasks.staged.jsonl + media/ to the laptop

# 4. Request dry run — one sanitized request per family, no API call, no ground truth
python mirl_ext/sft/gen_sft_targets.py --mode answer_blind_zero_shot \
    --tasks pilot_tasks.staged.jsonl --out data/sft/pilot_traces.jsonl \
    --image-root data/sft/media --dry-run

# 5. Pilot generation (laptop; 4 attempts/uid; NEVER run unasked — it bills)
python mirl_ext/sft/gen_sft_targets.py --mode answer_blind_zero_shot \
    --tasks pilot_tasks.staged.jsonl --out data/sft/pilot_traces.jsonl \
    --image-root data/sft/media

# 6. Resume after any interruption (same command; accepted/exhausted skip,
#    transport errors retry)

# 7. Pilot report + manual audit sample
python mirl_ext/sft/report_traces.py data/sft/pilot_traces.jsonl --audit 8

# 8. Parquet build (cluster; scp traces back first)
python mirl_ext/sft/build_sft_parquet.py --traces $DATA/split/pilot_traces.jsonl

# 9. Parquet validation + veRL one-batch load/collate smoke (cluster)
python mirl_ext/sft/smoke_sft_load.py --parquets $DATA/split_grpo/sft_parquet/*_sft.parquet

# 10. Full generation = steps 2-8 with --limit-per-family 2500 (or higher)

# 11. SFT training (engine trainer; no sbatch exists yet — see hazards below)
torchrun --nproc_per_node=8 -m verl.trainer.sft_trainer \
    model.path=$QWEN35 \
    data.train_files="[$DATA/split_grpo/sft_parquet/climb_train_sft.parquet,…]" \
    data.max_length=16384 data.truncation=error \
    trainer.total_epochs=1 trainer.project_name=mirl-sft
```

Local tests: `python mirl_ext/sft/tests/test_sft_pipeline.py` (or pytest).

## Known hazards before training

- **System turns**: `MultiTurnSFTDataset` tokenizes each message in isolation
  and Qwen3.5's template rejects a system-only list, so `build_sft_parquet.
  sft_messages()` merges the system turn into the user turn (train/serve
  difference vs. GRPO's true system turn; `smoke_sft_load.py` demonstrates the
  breakage and checks serve-prefix consistency).
- **Mixed media batches**: climb mixes image rows and video rows in one file;
  verl's `SFTTensorCollator` iterates the keys of `batch[0]`, so a batch mixing
  rows with and without `multi_modal_inputs` KeyErrors. All v1 rows carry
  media, but verify with the smoke test before long runs.
- **`data.max_length`** defaults to 1024 in `sft_trainer_engine.yaml` — far too
  small for these prompts + media tokens; set explicitly.
- **`<audio>`** in human_behaviour prompts has no audio column; it tokenizes as
  literal text (the transcript in the prompt stands in for audio).

## Native signals (blocked — do not train "on plots" thinking it's native)

`split_grpo` time-series rows ARE the rendered plots (`signals` was dropped by
`mirl_ext/rl/render_timeseries_images.py rewrite`; raw refs are recoverable via
`ts_images/<ds>_map.json`). A native-signal student path needs, exactly:
(1) an exporter from `alignment_state.pt["trainable_visual"]` back into a full
Qwen3.5 HF checkpoint (`model.visual.*`), (2) a dataset/processor mapping a
`signals` column to pseudo-video tensors + matching vision placeholders
(the render recipe exists as `mirl_ext/alignment/model.py::_timeseries_frames`
/ `_tactile_frames`), and (3) nothing model-side beyond stock video handling if
(2) uses the video path. Until then, ts-family SFT parquets train the plot
baseline, same as the existing GRPO indexes. The `*_tstok.parquet` files are a
separate signals-as-text A/B variant, unsplit and unused here.
