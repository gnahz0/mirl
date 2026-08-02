# MIRL Qwen3.5 migration ledger

This ledger is the gate between the historical MIRL fork and the Qwen3.5
migration.  The new branch starts from official `verl` commit
`6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`; no historical commit was merged or
cherry-picked.

## Provenance and safety snapshot

- New branch/worktree: `qwen3.5` at `/work/mit/ppliang_mit/alecz/mirl-qwen35`.
- Pinned base: `verl-project/verl@6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`.
- Historical fork point used for the 62-file inventory:
  `f56c89334f3d3b1942555782a10f28dee0bd2f28`.
- Historical baseline tip: `origin/baseline@24e208ede30bc067f03a061c29dda4eb79676056`.
- Reproduce the inventory with:

  ```bash
  git diff --name-only \
    f56c89334f3d3b1942555782a10f28dee0bd2f28..origin/baseline
  ```

  The result must contain exactly 62 paths.
- Pre-migration patches, source archives, repository state, and checksums are in
  `/work/mit/ppliang_mit/alecz/migration-backup/2026-07-20-qwen35/`.
  `SHA256SUMS` covers the backup files.  The 2.2 GB Stage-1 checkpoint was not
  duplicated; its source checksum is recorded separately.
- The `baseline` and `trainedve-raw` worktrees and the `alec-mv` environment are
  read-only migration inputs.

## Disposition vocabulary

- **ported**: present on `qwen3.5` now, with provenance retained.
- **superseded upstream**: intentionally not copied because current official
  `verl` owns the capability; re-open only if a targeted regression test fails.
- **advanced phase**: intentionally deferred to the named implementation stage,
  normally as a `mirl_ext` extension rather than an edit to upstream internals.
- **retired historical**: intentionally excluded from the new branch.

## Current migration status (2026-07-21)

- **Environment complete:** `environments/mirl-qwen35` records the working
  `alec-mv` prefix, exact Conda/Pip artifacts, build order, and CPU/B200 kernel
  verifier. Runtime and compiler caches are rooted on `/scratch`.
- **Combined image/video path complete:** `mirl_ext.data.MIRLDataset`, all six
  reward families, the deterministic eight-row real-media smoke fixture, and
  focused tests are ported. Historical general-purpose data builders not
  required by the existing Parquet indexes remain deferred.
- **Qwen3.5 combined trainer complete:**
  `examples/mirl/multiverse/run_qwen35_grpo.sh` and
  `examples/mirl/slurm/run_combined_b200.sbatch` replace the historical
  combined launcher. Two-B200 job `172783` completed all eight examples and 16
  rollouts through `training/global_step:1` with exit code 0.
- **One upstream regression was reopened:** current Qwen3.5 video prompts use
  one visual-token run per timestamped temporal patch, while returning one
  aggregate video grid. The focused adapter in
  `verl/experimental/agent_loop/agent_loop.py` expands only the position grid;
  its regression tests live in `tests/mirl/test_qwen35_mrope.py`.
- **Still deferred:** raw time-series actor inputs and Stage-1 visual alignment
  remain Stage 5. Their historical launchers are provenance snapshots, not
  runnable Qwen3.5 jobs.

## Historical baseline: all 62 paths

| # | Historical path | Disposition | Migration target or reason |
|---:|---|---|---|
| 1 | `.gitignore` | retired historical | Do not recreate the untracked `data` symlink; launchers will take an explicit data root. |
| 2 | `examples/multiverse_trainer/combined_qwen3_eval.sh` | ported | Replaced by the Qwen3.5 smoke mode in `examples/mirl/multiverse/run_qwen35_grpo.sh`. |
| 3 | `examples/multiverse_trainer/combined_qwen3_training.sh` | ported | Rebuilt as the Qwen3.5 FSDP2 + vLLM GRPO launcher. |
| 4 | `examples/multiverse_trainer/wait_for_gpus_and_run.sh` | advanced phase (Stage 4) | Preserve only if still useful beside Slurm allocation. |
| 5 | `examples/tactile/eval_qwen3_vl-8b-megatron_date.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 6 | `examples/tactile/eval_qwen3_vl-8b-megatron_glove.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 7 | `examples/tactile/eval_qwen3_vl-8b-megatron_question.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 8 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_date.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 9 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_date_no_tactile.sh` | retired historical | Obsolete ablation launcher. |
| 10 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_date_no_video.sh` | retired historical | Obsolete ablation launcher. |
| 11 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_date_single_view.sh` | retired historical | Obsolete ablation launcher. |
| 12 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_fail.sh` | retired historical | Obsolete one-off evaluation launcher. |
| 13 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_glove.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 14 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_question.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 15 | `examples/tactile/eval_qwen3_vl-8b-megatron_reasoning_task.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 16 | `examples/tactile/eval_qwen3_vl-8b-megatron_task.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 17 | `examples/tactile/eval_qwen3_vl-8b-megatron_vanilla_date.sh` | retired historical | Obsolete one-off evaluation launcher. |
| 18 | `examples/tactile/eval_qwen3_vl-8b-megatron_vanilla_other_splits.sh` | retired historical | Obsolete one-off evaluation launcher. |
| 19 | `examples/tactile/merge_sft_all.sh` | retired historical | The migration preserves instruction-model-to-GRPO, not the old SFT/Megatron path. |
| 20 | `examples/tactile/preprocess_sft_reasoning.py` | retired historical | Old SFT-only preprocessing is outside the active combined pipeline. |
| 21 | `examples/tactile/run_qwen3_vl-8b-megatron_date.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 22 | `examples/tactile/run_qwen3_vl-8b-megatron_glove.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 23 | `examples/tactile/run_qwen3_vl-8b-megatron_participant.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 24 | `examples/tactile/run_qwen3_vl-8b-megatron_question.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 25 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_all.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 26 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_date.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 27 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_date_2gpus.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 28 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_glove.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 29 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_question.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 30 | `examples/tactile/run_qwen3_vl-8b-megatron_reasoning_task.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 31 | `examples/tactile/run_qwen3_vl-8b-megatron_task.sh` | retired historical | Obsolete one-off Qwen3-VL/Megatron launcher. |
| 32 | `examples/tactile/run_qwen3_vl-8b-sft_all.sh` | retired historical | Old SFT launcher is outside the active instruction-model-to-GRPO workflow. |
| 33 | `scripts/build_parquet.py` | advanced phase (Stage 3) | Port schema-enforced unified Parquet building into `mirl_ext.data`. |
| 34 | `scripts/combine_datasets.py` | advanced phase (Stage 3) | Port the six-family combined builder into `mirl_ext.data`. |
| 35 | `scripts/count_video_tokens.py` | advanced phase (Stage 3) | Retain as a reusable schema/media validation tool. |
| 36 | `scripts/filter_by_token_limit.py` | advanced phase (Stage 3) | Rework around Qwen3.5 processor limits and the custom dataset. |
| 37 | `scripts/render_timeseries_images.py` | advanced phase (Stage 3) | Preserve the image time-series representation builder. |
| 38 | `scripts/smoke_dataset.py` | ported | Replaced by `mirl_ext.data.build_smoke` and modality tests. |
| 39 | `scripts/subsample_valid_fast.py` | advanced phase (Stage 3) | Preserve stratified validation subsampling. |
| 40 | `tests/reward_score/test_tactile.py` | ported | Rebuilt as parameterized `mirl_ext` reward tests. |
| 41 | `verl/experimental/agent_loop/agent_loop.py` | ported (focused regression) | Current Qwen3.5 timestamped videos reproduced a grid/run mismatch; carry only the tested position-grid adapter. |
| 42 | `verl/experimental/agent_loop/single_turn_agent_loop.py` | superseded upstream | Current multimodal rollout owns prompt/template construction. |
| 43 | `verl/experimental/reward_loop/reward_loop.py` | superseded upstream | Current custom reward path replaces global reward-loop dispatch changes. |
| 44 | `verl/model_merger/megatron_model_merger.py` | superseded upstream | Old Qwen3-VL/Megatron vision-weight merge patch is not part of the Qwen3.5 FSDP2 path. |
| 45 | `verl/models/mcore/registry.py` | superseded upstream | Current upstream has native Qwen3.5 model support; no Qwen3-VL registry patch. |
| 46 | `verl/protocol.py` | superseded upstream | Old reward metadata merge patch is dropped unless a focused mixed-source test fails. |
| 47 | `verl/trainer/config/data/legacy_data.yaml` | superseded upstream | Use current config plus `data.custom_cls`, not legacy global data keys. |
| 48 | `verl/trainer/constants_ppo.py` | advanced phase (Stage 3) | Keep decode limits local to MIRL media handling; avoid a global runtime-env edit. |
| 49 | `verl/trainer/ppo/ray_trainer.py` | advanced phase (Stage 3) | Add generic pooled classification metrics to the shared validation path. |
| 50 | `verl/utils/dataset/multiturn_sft_dataset.py` | superseded upstream | Active workflow is single-turn GRPO; retain no old SFT dataset fork. |
| 51 | `verl/utils/dataset/rl_dataset.py` | ported | Implemented through `mirl_ext.data.MIRLDataset`, selected by `data.custom_cls`. |
| 52 | `verl/utils/dataset/vision_utils.py` | advanced phase (Stage 3) | Move MIRL media policies to the extension and verify current upstream readers first. |
| 53 | `verl/utils/reward_score/__init__.py` | superseded upstream | Use `reward.custom_reward_function`; do not edit the global dispatcher. |
| 54 | `verl/utils/reward_score/ecg.py` | ported | Scoring lives in `mirl_ext.rewards.ecg`; pooled metrics remain separate. |
| 55 | `verl/utils/reward_score/haptic_ts.py` | ported | Scoring lives in `mirl_ext.rewards.haptic_ts`. |
| 56 | `verl/utils/reward_score/human_behaviour.py` | ported | Scoring lives in `mirl_ext.rewards.human_behaviour`. |
| 57 | `verl/utils/reward_score/medical.py` | ported | Medical/CLIMB scoring lives in `mirl_ext.rewards.medical`. |
| 58 | `verl/utils/reward_score/smellnet.py` | ported | Scoring lives in `mirl_ext.rewards.smellnet`. |
| 59 | `verl/utils/reward_score/tactile.py` | ported | Scoring lives in `mirl_ext.rewards.tactile`. |
| 60 | `verl/utils/tracking.py` | superseded upstream | Do not retain the old global W&B teardown patch unless reproduced. |
| 61 | `verl/workers/rollout/vllm_rollout/utils.py` | superseded upstream | Drop the old hashed ZMQ path patch unless the pinned stack reproduces the collision. |
| 62 | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | superseded upstream | Drop stale-socket mutation; current rollout internals and vLLM version differ. |

## Dirty baseline overlay

These five tracked edits were present beyond `origin/baseline` when the safety
snapshot was made.  They are recorded separately so the clean 62-file baseline
count remains reproducible.

| Dirty path | Disposition | Migration target or reason |
|---|---|---|
| `examples/multiverse_trainer/combined_qwen3_eval.sh` | advanced phase (Stage 4) | Carry useful override semantics into the Qwen3.5 launcher. |
| `examples/multiverse_trainer/combined_qwen3_training.sh` | advanced phase (Stage 4) | Carry the active six-family settings, image/token A/B mode, SP, offload, and bounded context settings. |
| `verl/models/transformers/qwen3_vl.py` | superseded upstream | Audit the dummy-visual-forward fix against native Qwen3.5; do not copy it. |
| `verl/trainer/ppo/ray_trainer.py` | advanced phase (Stage 3) | Generalize the dirty ECG macro-F1 implementation and test it on hand-computed predictions. |
| `verl/utils/reward_score/ecg.py` | advanced phase (Stage 3) | Preserve corrected parsing/scoring behavior in extension tests. |

Other untracked baseline inputs are also accounted for: `data` is a retired
symlink; `scripts/build_ts_token_parquet.py` advances to Stage 3;
`examples/multiverse_trainer/SESSION_NOTES.md` is absorbed as migration
provenance; and `trained-ve/final/{config.yaml,WRONG_DATA_DO_NOT_USE.md,
alignment_state.pt}` advance to the Stage-1 compatibility audit in Stage 5.

## Dirty raw-signal overlay

The `trainedve-raw` worktree at
`24e208ede30bc067f03a061c29dda4eb79676056` had the following 12 paths.  Every
path is **advanced phase (Stage 5)**; none is copied into upstream internals.

| Raw-signal path | Stage-5 target |
|---|---|
| `examples/multiverse_trainer/combined_qwen3_training.sh` | Qwen3.5 raw-signal run mode. |
| `verl/experimental/agent_loop/agent_loop.py` | Public multimodal passthrough or version-guarded adapter. |
| `verl/experimental/agent_loop/single_turn_agent_loop.py` | Public multimodal passthrough or version-guarded adapter. |
| `verl/models/transformers/qwen3_vl.py` | Qwen3.5 actor-input adapter; no Qwen3-VL monkey patch. |
| `verl/trainer/ppo/ray_trainer.py` | Generic validation metrics only; no raw-signal trainer fork. |
| `verl/utils/dataset/rl_dataset.py` | Raw schema through the same MIRL custom dataset class. |
| `verl/utils/reward_score/ecg.py` | Shared extension scorer. |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | vLLM 0.18 passthrough adapter after public-hook audit. |
| `scripts/build_raw_signal_parquets.py` | `mirl_ext.data` raw-signal builder. |
| `scripts/test_raw_signal_path.py` | Raw modality regression tests. |
| `verl/utils/dataset/signal_utils.py` | `mirl_ext.data.signals`. |
| `verl/workers/rollout/vllm_rollout/raw_signal_vllm.py` | Version-guarded `mirl_ext` vLLM adapter if still required. |

## Stage-1 alignment inventory

The 22 unique paths touched between `origin/engaging_hb_climb_tact` and
`origin/trained-ve` were originally classified as **advanced phase (Stage 5)**.
The runnable Qwen3.5 port now lives below `mirl_ext.alignment`; the old
`verl/trainer/alignment` subtree remains intentionally retired.

| Historical Stage-1 path | Stage-5 target |
|---|---|
| `examples/alignment/run_stage1_qwen3vl_clip.sh` | Qwen3.5 alignment launcher. |
| `examples/alignment/run_stage1_smoke.sh` | Bounded alignment smoke launcher. |
| `requirements.txt` | Replaced by the isolated Stage-2 environment specification and lock. |
| `scripts/build_climb_stratified.py` | `mirl_ext.data` builder. |
| `scripts/build_ecg_raw_jsonl.py` | `mirl_ext.data` raw builder. |
| `scripts/build_haptic_ts_jsonl.py` | `mirl_ext.data` raw builder. |
| `scripts/build_smellnet_raw_jsonl.py` | `mirl_ext.data` raw builder. |
| `scripts/clean_valid_jsonl.py` | `mirl_ext.data` validation utility. |
| `scripts/find_duplicates.py` | `mirl_ext.data` verification utility. |
| `scripts/split_combined_by_family.py` | `mirl_ext.data` family utility. |
| `scripts/verify_data.py` | Schema/media tests and CLI. |
| `scripts/verify_stage1_training.py` | Alignment parity and tensor-coverage tests. |
| `train_ve.sh` | Replaced by repository-owned Qwen3.5 alignment launchers. |
| `verl/trainer/alignment/__init__.py` | `mirl_ext.alignment`. |
| `verl/trainer/alignment/config/stage1_qwen3vl_clip.yaml` | Qwen3.5 Stage-1 config under the extension. |
| `verl/trainer/alignment/config/stage1_smoke.yaml` | Bounded Qwen3.5 smoke config. |
| `verl/trainer/alignment/data.py` | `mirl_ext.alignment.data`. |
| `verl/trainer/alignment/losses.py` | `mirl_ext.alignment.losses`. |
| `verl/trainer/alignment/model.py` | `mirl_ext.alignment.model`. |
| `verl/trainer/alignment/projection.py` | `mirl_ext.alignment.projection`. |
| `verl/trainer/alignment/trainer.py` | `mirl_ext.alignment.trainer`. |
| `verl/trainer/alignment/ts_renderer.py` | Shared raw-signal preprocessing under `mirl_ext`. |

Before any Stage-1 export, every required vision-tower key and shape must match
Qwen3.5-9B.  Existing Qwen3-VL artifacts are read-only and must never be
overwritten.

## Repository-owned Slurm launchers

The three root launchers were copied byte-for-byte into `examples/mirl/slurm/`
at bootstrap as provenance-controlled snapshots:

- `run_combined_b200.sbatch` (replaced by the runnable Stage-4 Qwen3.5 smoke),
- `run_trainedve_raw_b200.sbatch` (to be replaced in Stage 5), and
- `run_stage1_b200.sbatch` (to be replaced in Stage 5).

Only `run_combined_b200.sbatch` is Qwen3.5-ready. The Stage-1 snapshot still
names the removed `mirl-align` worktree; do not submit the raw-signal or Stage-1
snapshots. Their purpose is to make later launcher changes visible in branch
review.

## Deliberately dropped patch classes

The bootstrap initially carried no edits to upstream `verl`. Old
dummy-visual-forward, ZMQ socket-path, heterogeneous metadata, and dynamic-batch
patches remain classified as superseded upstream. The timestamped-video MRoPE
path was the one focused regression that failed on the pinned Qwen3.5 stack, so
that narrow adapter is now carried with tests. Checkpoint pruning remains absent
until a retention regression demonstrates a failure.
