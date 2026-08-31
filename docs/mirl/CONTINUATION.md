# MIRL Qwen3.5 continuation handoff

Last updated: 2026-07-21

> **FROZEN SNAPSHOT (2026-07-21) — superseded.** Current state, invariants,
> and launch paths live in `mirl_ext/CLAUDE.md`. Since this handoff: Stage-1
> alignment became the production pipeline (`mirl_ext/alignment/`, launcher
> `mirl_ext/alignment/run_stage1_b200.sbatch`), the SFT toolchain landed
> (`mirl_ext/sft/`), production `SMOKE=0` GRPO runs happened, a native
> time-series RL path exists (`TS_NATIVE=1`, `mirl_ext/rl/`), and the test
> suite grew past the counts quoted below. The deferred-work list and Slurm
> launcher census in this file no longer hold.

This is the operational handoff for the next agent working in
`$MIRL_CLUSTER_ROOT/mirl-qwen35`. Read it together with the repository
`AGENTS.md` instructions and `docs/mirl/README.md` before changing anything.

## Current outcome

The combined MIRL image/video GRPO path works with Qwen3.5-9B. The final exact
smoke launcher was verified by Slurm job `172783` on two NVIDIA B200 GPUs:

- Slurm state: `COMPLETED`, exit code `0`, elapsed time `00:03:47`.
- Eight real MIRL rows passed filtering: three image rows and five video rows.
- `rollout.n=2`, so the optimizer consumed 16 trajectories.
- `training/global_step:1` completed.
- `perf/total_num_tokens:25232`; prompt lengths ranged from 521 to 3,080.
- The B200 preflight passed FlashAttention, causal-conv1d, and FLA gated-delta
  CUDA kernels before the trainer started.
- The log is `$MIRL_CLUSTER_ROOT/logs/mirl_qwen35_172783.out`.

The current worktree changes are intentionally uncommitted. Do not discard or
reset them. Begin with `git status --short` and preserve unrelated user work.

## Session-start requirements

1. Read `$MIRL_CLUSTER_ROOT/AGENTS.md` if present and the instructions
   supplied for `$MIRL_CLUSTER_ROOT`.
2. Read `$MIRL_DATA_ROOT/SCRATCH_DATA.md`. Check its newest
   keep-alive date; run and record the touch pass only when it is more than 20
   days old. It was `2026-07-10` when this handoff was written, so no touch was
   required on 2026-07-21.
3. Keep large media, caches, compiler output, Ray sockets, and temporary files
   on `/scratch`. Do not use `/tmp`. Parquet/JSON indexes and source stay on
   `/work`, and media paths in those indexes point directly to `/scratch`.
4. Also read `data/TIMESERIES.md`, `data/TIME_SERIES_TOKENS.md`, and
   `$MIRL_CLUSTER_ROOT/mirl/examples/multiverse_trainer/SESSION_NOTES.md`
   for data provenance and historical Qwen3-VL behavior.

## Implemented files

- `mirl_ext/data/dataset.py`: `MIRLDataset`, selected with
  `data.custom_cls`. It normalizes JSON-string `extra_info`, removes phantom
  `<audio>` markers, handles nullable video dictionaries, enforces bounded
  image/video sizes, and decodes video asynchronously through TorchCodec.
- `mirl_ext/data/build_smoke.py`: builds a deterministic eight-row fixture at
  `$MIRL_DATA_ROOT/qwen35_smoke.parquet` from existing real
  MIRL data. Families are SmellNet, ECG, haptic time series, medical/CLIMB,
  human behaviour, and tactile.
- `mirl_ext/rewards/`: ports the six historical reward families. The combined
  dispatcher is `mirl_ext/rewards/combined.py`.
- `verl/experimental/agent_loop/agent_loop.py`: contains the only focused verl
  compatibility edit. `_grid_aligned_mm_token_types` expands Qwen3.5's one
  aggregate `[T,H,W]` video grid into `T` position-only `[1,H,W]` rows when the
  processor creates timestamp-separated video-token runs. It also avoids
  treating generated visual special tokens as new media.
- `tests/mirl/`: dataset, reward, and Qwen3.5 video-MRoPE regressions. Current
  result is 12 passed.
- `examples/mirl/multiverse/run_qwen35_grpo.sh`: Qwen3.5-9B FSDP2/vLLM GRPO
  launcher. `SMOKE=1` is bounded to eight rows, two rollouts, 128 response
  tokens, and one optimizer step.
- `examples/mirl/slurm/run_combined_b200.sbatch`: runnable two-B200 smoke job.
  (Since this snapshot the directory changed: `run_trainedve_raw_b200.sbatch`
  remains a historical Qwen3-VL provenance snapshot — do not submit — and the
  SFT launcher shim execs `mirl_ext/sft/`; see `examples/mirl/slurm/README.md`.)
- `environments/mirl-qwen35/`: builder, verifier, human specification, and
  exact Conda/Pip locks for `mirl-b200`.
- `docs/mirl/qwen35-migration-ledger.md`: historical file-by-file disposition
  and the boundary between completed and deferred migration stages.

## Environment state

Active prefix:

```text
$MIRL_PYENV
```

Important verified pins:

- Python 3.12.13
- PyTorch 2.10.0+cu129, torchvision 0.25.0+cu129, torchaudio 2.10.0+cu129
- vLLM 0.18.0
- Transformers 5.3.0.dev0 at
  `cc7ab9be508ce6ed3637bba9e50367b29b742dc6`
- FlashAttention 2.8.3
- Flash Linear Attention 0.5.1 at
  `c525f4957f11a6f197b52c0c222377446c3eab56`
- causal-conv1d 1.6.2.post1
- nvidia-cutlass-dsl 4.4.2 and quack-kernels 0.3.4
- qwen-vl-utils 0.0.14, TorchCodec 0.10.0, FFmpeg 7.1.1
- TransferQueue 0.1.8
- CUDA NVRTC 12.9.86 and cuBLAS development headers 12.9.1.4
- editable verl checkout based on
  `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`

The CUTLASS/Quack pair is deliberate. Newer CUTLASS 4.6.1 with Quack 0.5.0
failed vLLM 0.18 because the expected `ThrMma` interface was absent. The CUDA
development headers are also required for FlashInfer JIT compilation.

`pip check` has exactly one accepted metadata diagnostic: vLLM 0.18 declares
Transformers `<5`, while the Qwen3.5-tested Transformers commit reports itself
as 5.3.0.dev0. The environment verifier checks that this is the only issue and
then directly verifies the actual Qwen3.5 and vLLM imports.

## Commands to reproduce the checks

Run unit tests from the Qwen3.5 worktree:

```bash
cd $MIRL_CLUSTER_ROOT/mirl-qwen35
export PYTHONNOUSERSITE=1
export TMPDIR=$MIRL_SCRATCH_ROOT/tmp-qwen35/manual
"$MIRL_PYENV"/bin/python -m pytest -q tests/mirl
```

Run the CPU/import/model-snapshot verifier:

```bash
export PIP_CACHE_DIR=$MIRL_SCRATCH_ROOT/pip-cache-qwen35
export XDG_CACHE_HOME=$MIRL_SCRATCH_ROOT/cache-qwen35/xdg
export TRITON_CACHE_DIR=$MIRL_SCRATCH_ROOT/cache-qwen35/triton
export TORCHINDUCTOR_CACHE_DIR=$MIRL_SCRATCH_ROOT/cache-qwen35/inductor
export VLLM_CACHE_ROOT=$MIRL_SCRATCH_ROOT/cache-qwen35/vllm
export FLASHINFER_WORKSPACE_BASE=$MIRL_SCRATCH_ROOT/cache-qwen35/flashinfer

"$MIRL_PYENV"/bin/python \
  environments/mirl-qwen35/verify_environment.py \
  --model-snapshot \
  $MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

Submit the full GPU preflight plus one-step smoke:

```bash
cd $MIRL_CLUSTER_ROOT/mirl-qwen35
sbatch examples/mirl/slurm/run_combined_b200.sbatch
```

The job writes `logs/mirl_qwen35_<job-id>.out` under
`$MIRL_CLUSTER_ROOT`. Success requires all of the following, not just
a zero exit code:

```text
cuda_device=NVIDIA B200
train dataset size: 8, val dataset size: 8
Training Progress: 100% ... 1/1
training/global_step:1
perf/total_num_tokens:25232
```

There must be no `StopIteration` or `Error in _run_prompt`. With eight rows and
two rollouts, `25232 == 16 * (1449 average prompt tokens + 128 response tokens)`
confirms that video rows were not silently dropped.

## Why the video adapter exists

The first full model attempt, job `172737`, reached vLLM and completed an
optimizer step but five video rows failed in post-rollout position rebuilding:

```text
Qwen3VLModel.get_rope_index -> next(grid_iters[2]) -> StopIteration
```

The processor emitted four timestamp-separated video-token runs for a sample
with `video_grid_thw=[[4,H,W]]`. The model helper consumed one grid per run.
The adapter expands only the position grid to four `[1,H,W]` rows; the original
aggregate grid is retained for the vision encoder. Jobs `172766`, `172775`, and
the final exact job `172783` all completed the full 16-trajectory batch after
this fix.

## Known non-blocking log noise

- vLLM warns that `mrope_section` and `mrope_interleaved` are unrecognized in
  `rope_parameters`; rollout and the Transformers training forward still agree
  closely enough for the smoke (`training/rollout_probs_diff_valid:1.0`).
- Python's multiprocessing resource tracker may print a shared-memory `KeyError`
  when vLLM sleeps or tears down workers. Job `172783` logged this but completed
  the optimizer step and exited 0. Do not confuse it with the fixed video
  `StopIteration`.
- The launcher now sets smoke filtering to single-process and DataLoader workers
  to zero. This removed the earlier `/scratch` NFS `.nfs... resource busy` and
  DataLoader-shutdown tracebacks.
- FlashAttention may warn during initial model construction that the temporary
  dtype is float32; the loaded training and rollout weights run in bfloat16.

## Deferred work and safe next steps

The validated scope is the combined image/video representation. These remain
deferred and should not be inferred as working:

1. Raw time-series tokens as direct actor inputs (`TS_TOKENS=1` only selects
   existing token Parquets; it is not the historical raw-signal model adapter).
2. Stage-1 visual alignment from `origin/trained-ve` and the historical
   alignment checkpoint.
3. A multi-step or production-length run with `SMOKE=0`.
4. Combined evaluation beyond the one-step trainer smoke.

The safest next validation is a short 2-3-step combined run using the same
image/video representation and B200 stack. Keep `use_remove_padding=False` and
Ulysses sequence parallel size `1`: Qwen3.5-9B interleaves Gated Delta Net and
full-attention layers, and the packed-sequence GDN path has not been validated.
Do not submit `examples/mirl/slurm/run_trainedve_raw_b200.sbatch`; it is a
historical Qwen3-VL provenance snapshot. (The old Stage-1 snapshot of the same
name was removed; Stage-1 now submits the production launcher
`mirl_ext/alignment/run_stage1_b200.sbatch` — see `mirl_ext/CLAUDE.md`.)

Before committing, rerun the tests, environment verifier, `bash -n` on the
three runnable shell files, and `git diff --check`. No commit has been requested
or created as of this handoff.
