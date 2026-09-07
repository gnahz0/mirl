# MIRL Qwen3.5 environment

This directory records the B200-specific environment used by the Qwen3.5
migration. The active prefix is `$MIRL_PYENV`; use the build script below to
create an isolated test prefix.

## Pinned stack

| Component | Pin |
| --- | --- |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.10.0+cu129 |
| torchvision / torchaudio | 0.25.0+cu129 / 2.10.0+cu129 |
| vLLM | 0.18.0 |
| Transformers | `cc7ab9be508ce6ed3637bba9e50367b29b742dc6` |
| FlashAttention | 2.8.3, compiled for B200 `sm100` |
| Flash Linear Attention | 0.5.1 at `c525f4957f11a6f197b52c0c222377446c3eab56` |
| causal-conv1d | 1.6.2.post1, official CUDA 12 / Torch 2.10 wheel |
| CUTLASS DSL / Quack | 4.4.2 / 0.3.4 (vLLM 0.18 release-compatible pair) |
| qwen-vl-utils | 0.0.14 |
| TorchCodec / FFmpeg | 0.10.0 / 7.1.1 |
| CUDA JIT headers | NVRTC 12.9.86 / cuBLAS 12.9.1.4 |
| TransferQueue | 0.1.8 |
| verl | editable checkout based on `6a6242f3d8ec7d9f8b4936f4905144707d91fe3b` |

`environment.yml` is the human-maintained Conda specification.
`conda-linux-64.lock` is the exact Linux artifact lock from the working prefix.
`pip-freeze.lock` is the resolved pip inventory; it is an audit record rather
than a single-shot requirements file because vLLM 0.18 declares Transformers
`<5` while the upstream-tested Qwen3.5 commit reports version `5.3.0.dev0`.
`build_cluster_env.sh` installs those packages in the tested order.

## Build and verify

Request an interactive B200 node, then run:

```bash
srun --partition=b200-devel --account=$SBATCH_ACCOUNT --nodes=1 \
  --gpus-per-node=1 --cpus-per-task=8 --mem=256G --time=04:00:00 \
  --job-name=mirl-env-build --pty bash -l

MIRL_ENV_PREFIX=$MIRL_PYENV-qwen35-test \
  $MIRL_CLUSTER_ROOT/mirl-qwen35/environments/mirl-qwen35/build_cluster_env.sh
```

The builder refuses to overwrite an existing prefix. Pip, compiler, Triton,
TorchInductor, and vLLM caches are all placed below
`$MIRL_SCRATCH_ROOT`; Python user-site packages are disabled. It uses the
cluster GCC 11.5 for FlashAttention because Conda's CUDA headers live in a
target-specific include directory and the cluster compiler is the tested host
compiler for this wheel.

The current prefix can be checked without downloads using:

```bash
export PYTHONNOUSERSITE=1
export TMPDIR=$MIRL_SCRATCH_ROOT/tmp-qwen35
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

Add `--cuda` on a B200 node. The verifier accepts exactly one `pip check`
diagnostic: vLLM's stale Transformers upper bound. It separately proves the
Qwen3.5 Transformers class, vLLM registry entry, local processor, TorchCodec
video backend, vLLM's CUTLASS import, FlashAttention, causal-conv1d, and FLA
gated-delta kernels work with the pinned combination.
