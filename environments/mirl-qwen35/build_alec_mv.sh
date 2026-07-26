#!/usr/bin/env bash
set -euo pipefail

# Build this environment from a B200 interactive Slurm allocation. The default
# paths are the cluster locations used by MIRL, but all three may be overridden.
MIRL_ENV_PREFIX="${MIRL_ENV_PREFIX:-/work/mit/ppliang_mit/alecz/envs/alec-mv}"
MIRL_WORKTREE="${MIRL_WORKTREE:-/work/mit/ppliang_mit/alecz/mirl-qwen35}"
MIRL_SCRATCH_ROOT="${MIRL_SCRATCH_ROOT:-/scratch/dvdai_mit/alecz}"
MIRL_HF_CACHE_ROOT="${MIRL_HF_CACHE_ROOT:-/work/mit/ppliang_mit/alecz/hf_cache}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this script inside an interactive Slurm allocation." >&2
  exit 2
fi
if [[ ! -d "${MIRL_WORKTREE}/verl" ]]; then
  echo "MIRL_WORKTREE is not a verl checkout: ${MIRL_WORKTREE}" >&2
  exit 2
fi
if [[ -e "${MIRL_ENV_PREFIX}" ]]; then
  echo "Refusing to overwrite existing path: ${MIRL_ENV_PREFIX}" >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '^NVIDIA B200$'; then
  echo "The FlashAttention wheel in this environment is intentionally B200-only." >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="${MIRL_SCRATCH_ROOT}/pip-cache-qwen35"
export TMPDIR="${MIRL_SCRATCH_ROOT}/tmp-qwen35-slurm-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${MIRL_SCRATCH_ROOT}/cache-qwen35/xdg"
export TRITON_CACHE_DIR="${MIRL_SCRATCH_ROOT}/cache-qwen35/triton"
export TORCHINDUCTOR_CACHE_DIR="${MIRL_SCRATCH_ROOT}/cache-qwen35/inductor"
export VLLM_CACHE_ROOT="${MIRL_SCRATCH_ROOT}/cache-qwen35/vllm"
export FLASHINFER_WORKSPACE_BASE="${MIRL_SCRATCH_ROOT}/cache-qwen35/flashinfer"
export HF_HOME="${MIRL_HF_CACHE_ROOT}"
mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}"

conda create --yes --prefix "${MIRL_ENV_PREFIX}" \
  --channel nvidia --channel defaults \
  python=3.12.13 \
  pip=26.1.1 \
  setuptools=80.9.0 \
  wheel=0.47.0 \
  cmake=4.2.3 \
  cuda-nvcc=12.9.86 \
  cuda-cudart-dev=12.9.79 \
  cuda-nvrtc-dev=12.9.86 \
  libcublas-dev=12.9.1.4 \
  gcc_linux-64=14.3.0 \
  gxx_linux-64=14.3.0

# TorchCodec needs shared FFmpeg libraries. Install this after the core toolchain
# so adding conda-forge does not change the already pinned compiler/CUDA solve.
conda install --yes --prefix "${MIRL_ENV_PREFIX}" \
  --channel conda-forge --channel nvidia --channel defaults \
  ffmpeg=7.1.1

MIRL_PYTHON="${MIRL_ENV_PREFIX}/bin/python"
"${MIRL_PYTHON}" -m pip install --no-input \
  ninja==1.13.0 packaging==25.0
"${MIRL_PYTHON}" -m pip install --no-input \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# Install vLLM's declared dependencies before applying the exact Qwen3.5
# Transformers revision. vLLM 0.18's metadata says Transformers <5, while the
# upstream-tested Qwen3.5 revision reports itself as 5.3.0.dev0.
"${MIRL_PYTHON}" -m pip install --no-input \
  vllm==0.18.0 qwen-vl-utils==0.0.14 torchcodec==0.10.0 TransferQueue==0.1.8 \
  nvidia-cutlass-dsl-libs-base==4.4.2 nvidia-cutlass-dsl==4.4.2 \
  quack-kernels==0.3.4 \
  uvicorn==0.51.0 pytest==9.1.1
"${MIRL_PYTHON}" -m pip install --no-input --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
"${MIRL_PYTHON}" -m pip install --no-input \
  "transformers @ git+https://github.com/huggingface/transformers.git@cc7ab9be508ce6ed3637bba9e50367b29b742dc6"
"${MIRL_PYTHON}" -m pip install --no-input --no-build-isolation \
  "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@c525f4957f11a6f197b52c0c222377446c3eab56"

# Conda stores CUDA headers below targets/x86_64-linux. FlashAttention's host
# C++ compilation does not discover that directory without these variables.
export CUDA_HOME="${MIRL_ENV_PREFIX}"
export PATH="${MIRL_ENV_PREFIX}/bin:${PATH}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=100
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
"${MIRL_PYTHON}" -m pip install --no-input --no-build-isolation flash-attn==2.8.3

"${MIRL_PYTHON}" -m pip install --no-input --editable "${MIRL_WORKTREE}"
"${MIRL_PYTHON}" "${MIRL_WORKTREE}/environments/mirl-qwen35/verify_environment.py" --cuda
