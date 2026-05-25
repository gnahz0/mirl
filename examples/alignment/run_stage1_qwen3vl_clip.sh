#!/usr/bin/env bash
# Stage 1 multimodal alignment launcher (single GPU).
#
# Usage:
#   bash examples/alignment/run_stage1_qwen3vl_clip.sh
#   # or with overrides:
#   bash examples/alignment/run_stage1_qwen3vl_clip.sh train.batch_size=8 train.total_steps=500
#
# Env vars honored by the existing repo (kept consistent with combined_qwen3_training.sh):
#   QWEN_VL_MAX_IMAGE_TOKENS   - cap per-image visual tokens (default 16384, we set 1024)
#   MIRL_ROOT                  - repo root used by the YAML for data path interpolation
#   CUDA_VISIBLE_DEVICES       - pick the GPU
#
# TODO(stage2): once we have a trained checkpoint, run an export script that drops
# `trainable_visual` weights back into a Qwen3-VL HF checkpoint dir, then point
# combined_qwen3_training.sh's actor_rollout_ref.model.path at it.

set -e

MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRL_ROOT

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export QWEN_VL_MAX_IMAGE_TOKENS="${QWEN_VL_MAX_IMAGE_TOKENS:-1024}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Forward W&B env so the trainer can pick up creds without touching the YAML.
# Set WANDB_API_KEY before launching to log online; otherwise the trainer falls back
# to WANDB_MODE=offline and writes runs to ./wandb/.
export WANDB_PROJECT="${WANDB_PROJECT:-mirl-alignment}"
export WANDB_DIR="${WANDB_DIR:-${MIRL_ROOT}/outputs/alignment_stage1/wandb}"
mkdir -p "${WANDB_DIR}"

CONFIG="${MIRL_ROOT}/verl/trainer/alignment/config/stage1_qwen3vl_clip.yaml"

mkdir -p "${MIRL_ROOT}/outputs/alignment_stage1"

set -x
python3 -m verl.trainer.alignment.trainer \
    --config "${CONFIG}" \
    "$@"
