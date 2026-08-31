# MIRL on Qwen3.5

For an agent-to-agent operational handoff with exact job history, environment
pins, reproduction commands, and deferred work, read
[`CONTINUATION.md`](CONTINUATION.md) first.

This worktree ports MIRL's multimodal reinforcement-learning pipeline from the
historical Qwen3-VL fork to Qwen3.5. It is based on official verl commit
`6a6242f3d8ec7d9f8b4936f4905144707d91fe3b`; MIRL-specific behavior lives in
`mirl_ext` instead of patching verl internals.

## What is implemented

- `mirl_ext.data.MIRLDataset` adapts the existing Parquet files to current
  verl, including JSON-encoded `extra_info`, historical phantom audio markers,
  TorchCodec video decoding, and bounded image/video token policies.
- `mirl_ext.rewards.combined.compute_score` dispatches the six MIRL reward
  families: tactile, human behaviour, medical/CLIMB, SmellNet, ECG, and haptic
  time series.
- `examples/mirl/multiverse/run_qwen35_grpo.sh` is the Qwen3.5-9B FSDP2 + vLLM
  GRPO launcher. `SMOKE=1` reduces it to eight real-media examples and one
  optimizer step.
- `verl/experimental/agent_loop/agent_loop.py` contains the one narrow upstream
  compatibility fix: Qwen3.5 expands each video into timestamp-separated token
  runs, so position-only video grids must be expanded to the same temporal
  layout. A regression also prevents generated visual special tokens from
  consuming nonexistent media grids.
- `environments/mirl-qwen35` defines and verifies the B200-specific `mirl-b200`
  stack. Temporary files, pip downloads, Ray sockets, and compiler caches live
  on `/scratch`, while source, logs, and Parquet indexes remain on `/work`.

The default GRPO pipeline still uses the existing image/video representation for
all six families. Raw-signal Stage-1 visual alignment is now runnable as a
separate Qwen3.5 pipeline under `mirl_ext.alignment`; exporting its trained visual
tower into a full Qwen checkpoint for SFT/RL remains the next handoff step.

Qwen3.5-9B interleaves Gated Delta Net and full-attention layers. The launcher
therefore keeps padding removal and Ulysses sequence parallelism disabled, in
line with the current upstream Qwen3.5 video recipe, while retaining FSDP2 for
training sharding.

## Repository map

| Path | Purpose |
| --- | --- |
| `mirl_ext/data/dataset.py` | Current verl dataset adapter and media limits |
| `mirl_ext/data/build_smoke.py` | Deterministic eight-row fixture from real MIRL data |
| `mirl_ext/rewards/` | Six reward implementations and combined dispatcher |
| `mirl_ext/alignment/` | Qwen3.5 vision/SigLIP2 raw-signal alignment pipeline |
| `examples/mirl/multiverse/run_qwen35_grpo.sh` | Local/allocation launcher |
| `examples/mirl/slurm/run_combined_b200.sbatch` | Two-B200 one-step smoke submission |
| `environments/mirl-qwen35/` | Rebuild script, verifier, and exact environment records |
| `tests/mirl/` | Dataset, reward, and Qwen3.5 video-MRoPE regressions |
| `docs/mirl/CONTINUATION.md` | Frozen 2026-07-21 migration handoff (current state: `mirl_ext/CLAUDE.md`) |
| `docs/mirl/qwen35-migration-ledger.md` | Historical file-by-file migration decisions |

## Verify `mirl-b200`

From the repository root:

```bash
export PYTHONNOUSERSITE=1
export TMPDIR=$MIRL_SCRATCH_ROOT/tmp-qwen35
export PIP_CACHE_DIR=$MIRL_SCRATCH_ROOT/pip-cache-qwen35
export FORCE_QWENVL_VIDEO_READER=torchcodec

"$MIRL_PYENV"/bin/python \
  environments/mirl-qwen35/verify_environment.py \
  --model-snapshot \
  $MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

Use `--cuda` inside a B200 allocation to exercise FlashAttention,
causal-conv1d, and the FLA gated-delta kernel.

## Run the smoke test

```bash
cd $MIRL_CLUSTER_ROOT/mirl-qwen35
sbatch examples/mirl/slurm/run_combined_b200.sbatch
```

The default submission requests two B200s and performs one GRPO step. A full
run is deliberately opt-in with `SMOKE=0`; tune the full-run resource and
context settings only after the smoke remains green.

## Verified result

On 2026-07-21, Slurm job `172783` completed with exit code 0 on two B200s. It
used all eight fixture rows (three image and five video examples), generated
two rollouts per row, updated the actor once (`training/global_step:1`), and
processed 25,232 tokens. The environment preflight exercised FlashAttention,
causal-conv1d, and the FLA gated-delta kernel on the B200 before training.
