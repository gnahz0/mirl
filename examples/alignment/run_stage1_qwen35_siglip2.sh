#!/usr/bin/env bash
# Stage 1 multimodal alignment launcher.
#
# NUM_GPUS>1 shards the batch across GPUs with throughput DDP (launched via
# torchrun; the trainer auto-detects WORLD_SIZE). Edit the USER SETTINGS block
# below, then just run it:
#
# Usage:
#   ./examples/alignment/run_stage1_qwen35_siglip2.sh
#   ./examples/alignment/run_stage1_qwen35_siglip2.sh train.total_steps=500
#   NUM_GPUS=1 ./examples/alignment/run_stage1_qwen35_siglip2.sh
#
# DDP note: global batch = train.batch_size * grad_accum_steps * NUM_GPUS. The
# contrastive loss uses each rank's LOCAL negatives only (this buys throughput,
# not a larger negative pool).
# TODO(stage2): once we have a trained checkpoint, run an export script that drops
# `trainable_visual` weights back into a Qwen3.5 HF checkpoint dir, then point
# the Qwen3.5 RL launcher's actor_rollout_ref.model.path at it.

set -e

# ============================== USER SETTINGS ==============================
# Edit these defaults; each can still be overridden by an env var of the same
# name (e.g. `NUM_GPUS=1 ./run_...sh`).
: "${NUM_GPUS:=2}"                       # GPUs to shard across (1 = single-GPU, >1 = torchrun DDP)
: "${QWEN_VL_MAX_IMAGE_TOKENS:=512}"     # per-image visual tokens (1024 OOMs on H200)
: "${VIDEO_MAX_FRAMES:=8}"               # frames sampled per video (8 = standard; faster steps)
# CUDA_VISIBLE_DEVICES: leave unset to auto-pick 0..NUM_GPUS-1, or pin explicitly.
# ===========================================================================

MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRL_ROOT
# cd so ``python -m mirl_ext.alignment.trainer`` can find the package on
# sys.path regardless of where the user invoked the launcher from. Also so
# any relative paths in the config (e.g. ``./data/...``) resolve correctly.
cd "${MIRL_ROOT}"
export PYTHONPATH="${MIRL_ROOT}:${PYTHONPATH:-}"

# Default device list: GPU 0 for single-GPU, or 0..NUM_GPUS-1 for DDP.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  else
    CUDA_VISIBLE_DEVICES=0
  fi
fi
export CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Set in USER SETTINGS above; exported here for the trainer + qwen_vl_utils.
export QWEN_VL_MAX_IMAGE_TOKENS
export VIDEO_MAX_FRAMES
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export TORCHCODEC_LOG_LEVEL="${TORCHCODEC_LOG_LEVEL:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# Reduce CUDA allocator fragmentation -- helps larger batch sizes fit on the H200.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# W&B env (online if WANDB_API_KEY is set; otherwise trainer falls back to offline).
export WANDB_PROJECT="${WANDB_PROJECT:-mirl-alignment}"
export WANDB_DIR="${WANDB_DIR:-${MIRL_ROOT}/outputs/alignment_stage1/wandb}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${MIRL_ROOT}/outputs/alignment_stage1"

CONFIG="${MIRL_ROOT}/mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Launching ${NUM_GPUS}-GPU DDP via torchrun (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    set -x
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
        -m mirl_ext.alignment.trainer \
        --config "${CONFIG}" \
        "$@"
else
    set -x
    python3 -u -m mirl_ext.alignment.trainer \
        --config "${CONFIG}" \
        "$@"
fi
