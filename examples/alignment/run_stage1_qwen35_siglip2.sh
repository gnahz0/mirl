#!/usr/bin/env bash
# Stage-1 alignment launcher. Configuration overrides follow the config path.
#
# Usage:
#   ./examples/alignment/run_stage1_qwen35_siglip2.sh
#   ./examples/alignment/run_stage1_qwen35_siglip2.sh train.num_train_epochs=2
#   NUM_GPUS=1 ./examples/alignment/run_stage1_qwen35_siglip2.sh
#
# Global batch = train.batch_size * grad_accum_steps * NUM_GPUS.

set -eo pipefail

: "${NUM_GPUS:=2}"

MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRL_ROOT
cd "${MIRL_ROOT}"
export PYTHONPATH="${MIRL_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
export CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export TORCHCODEC_LOG_LEVEL="${TORCHCODEC_LOG_LEVEL:-0}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export WANDB_PROJECT="${WANDB_PROJECT:-mirl-alignment}"
export WANDB_DIR="${WANDB_DIR:-${MIRL_ROOT}/outputs/alignment_stage1/wandb}"
mkdir -p "${WANDB_DIR}"

: "${CONFIG:=${MIRL_ROOT}/mirl_ext/alignment/config/stage1_qwen35_siglip2_aicr.yaml}"

echo "Launching ${NUM_GPUS} GPU(s) via torchrun (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
set -x
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    -m mirl_ext.alignment.trainer \
    --config "${CONFIG}" \
    "$@"
