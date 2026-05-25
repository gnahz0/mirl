#!/usr/bin/env bash
# Stage 1 multimodal alignment SMOKE TEST -- single GPU, ~3-10 minutes.
#
# Validates that the whole pipeline (dataset -> Qwen3-VL VE forward -> CLIP text ->
# projections -> losses -> backprop -> W&B logging -> checkpointing) actually works
# end-to-end on a tiny slice of data before you commit to the full run.
#
# Differences from run_stage1_qwen3vl_clip.sh (the full launcher):
#   * 200 stratified samples from combined_valid_mini.json instead of 50k from full.
#   * batch_size 4 / grad_accum 1 instead of 16 / 4.
#   * 20 steps with warmup 5 (vs 5000 / 200).
#   * log every step, checkpoint at steps 10 and 20.
#   * W&B is ON (run name appended with -smoke and tagged "smoke" so it's easy to filter).
#
# Usage:
#   bash examples/alignment/run_stage1_smoke.sh

set -e

MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRL_ROOT

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export QWEN_VL_MAX_IMAGE_TOKENS="${QWEN_VL_MAX_IMAGE_TOKENS:-1024}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
# Same defensive video defaults as the rest of the repo (see vision_utils.py).
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export TORCHCODEC_LOG_LEVEL="${TORCHCODEC_LOG_LEVEL:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# W&B settings (online if WANDB_API_KEY is set; otherwise the trainer falls back to offline).
export WANDB_PROJECT="${WANDB_PROJECT:-mirl-alignment}"
export WANDB_DIR="${WANDB_DIR:-${MIRL_ROOT}/outputs/alignment_stage1/wandb}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${MIRL_ROOT}/outputs/alignment_stage1"

CONFIG="${MIRL_ROOT}/verl/trainer/alignment/config/stage1_qwen3vl_clip.yaml"

set -x
python3 -m verl.trainer.alignment.trainer \
    --config "${CONFIG}" \
    data.train_files="[${MIRL_ROOT}/data/combined_valid_mini.json]" \
    data.max_train_samples=200 \
    data.balanced_sampling_key=data_source \
    train.batch_size=4 \
    train.grad_accum_steps=1 \
    train.num_workers=2 \
    train.total_steps=20 \
    train.warmup_steps=5 \
    train.log_every=1 \
    train.ckpt_every=10 \
    wandb.enable=true \
    wandb.name=stage1_qwen3vl_clip-smoke \
    'wandb.tags=[stage1,qwen3vl,clip-vit-large-patch14,smoke]'
