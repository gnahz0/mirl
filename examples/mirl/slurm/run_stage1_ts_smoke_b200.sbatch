#!/bin/bash
# Focused two-step gate: force multi-sample scalar/tactile batches so the
# SigLIP2 label-alignment (InfoNCE) path is non-degenerate.
#SBATCH --job-name=mirl_s1_ts_smoke
#SBATCH --partition=b200-batch
#SBATCH --account=ppliang_mit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/work/mit/ppliang_mit/alecz/logs/align_stage1_ts_smoke_%j.out
#SBATCH --error=/work/mit/ppliang_mit/alecz/logs/align_stage1_ts_smoke_%j.out

set -eo pipefail

ROOT=/work/mit/ppliang_mit/alecz
cd "$ROOT/mirl-qwen35"

export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/mirl-qwen35:${PYTHONPATH:-}"
export QWEN_VL_MAX_IMAGE_TOKENS=512
export VIDEO_MAX_FRAMES=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/envs/alec-mv"

python -u -m mirl_ext.alignment.trainer \
  --config mirl_ext/alignment/config/stage1_smoke.yaml \
  'data.data_source_filter=[smellnet_base,smellnet_mixture,ecg,haptic_tactile]' \
  train.total_steps=2 \
  train.val_every=0 \
  train.output_dir=/scratch/dvdai_mit/alecz/checkpoints/alignment_ts_smoke_unified_grayscale_clean_prelaunch_20260725
