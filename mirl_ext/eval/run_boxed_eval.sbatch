#!/bin/bash
# Offline greedy boxed-answer eval of one model over the GRPO val parquets (one B200).
# Base and SFT arms are two submissions of this script with different MODEL_PATH:
#   source mirl.env && MODEL_PATH=$MIRL_SCRATCH_ROOT/models/qwen35-p20-sft \
#     sbatch --output "$MIRL_SCRATCH_ROOT/logs/boxed_eval_%j.out" mirl_ext/eval/run_boxed_eval.sbatch
#   MODEL_PATH=$MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/<rev>  # base arm
# Extra args pass through to run_boxed_eval.py (e.g. --families ecg_valid --max-samples 32).
#SBATCH -J mirl-boxed-eval
#SBATCH -p b200-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --exclude=b0028,b0024,b0012

source $MIRL_CONDA_SH
conda activate $MIRL_PYENV
set -euo pipefail

MIRL_ROOT=$MIRL_CLUSTER_ROOT/mirl-qwen35
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to an HF model dir}
DATA_ROOT=${DATA_ROOT:-$MIRL_DATA_ROOT}
EVAL_NAME=${EVAL_NAME:-$(basename "$MODEL_PATH")-${SLURM_JOB_ID:-manual}}
OUT_DIR=${OUT_DIR:-$MIRL_SCRATCH_ROOT/eval/boxed}

export PYTHONNOUSERSITE=1
export PYTHONPATH=$MIRL_ROOT
export HF_HOME=$MIRL_CLUSTER_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=torchcodec
export DECORD_EOF_RETRY_MAX=50
export TMPDIR=$MIRL_SCRATCH_ROOT/tmp-qwen35/${SLURM_JOB_ID:-manual}
export XDG_CACHE_HOME=$MIRL_SCRATCH_ROOT/cache-qwen35/xdg
export TRITON_CACHE_DIR=$MIRL_SCRATCH_ROOT/cache-qwen35/triton
export TORCHINDUCTOR_CACHE_DIR=$MIRL_SCRATCH_ROOT/cache-qwen35/inductor
export VLLM_CACHE_ROOT=$MIRL_SCRATCH_ROOT/cache-qwen35/vllm
export FLASHINFER_WORKSPACE_BASE=$MIRL_SCRATCH_ROOT/cache-qwen35/flashinfer
# CUDA toolchain lives in the conda env; vLLM/flashinfer JIT needs it (run_qwen35_grpo.sh parity).
export CUDA_HOME=${CUDA_HOME:-$MIRL_PYENV}
export CPATH=$MIRL_PYENV/targets/x86_64-linux/include${CPATH:+:$CPATH}
export CPLUS_INCLUDE_PATH=$MIRL_PYENV/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
export LIBRARY_PATH=$MIRL_PYENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$MIRL_PYENV/targets/x86_64-linux/lib:$MIRL_PYENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$OUT_DIR"

cd "$MIRL_ROOT"
python -m mirl_ext.eval.run_boxed_eval \
    --model "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --out "$OUT_DIR/$EVAL_NAME.json" \
    "$@"
