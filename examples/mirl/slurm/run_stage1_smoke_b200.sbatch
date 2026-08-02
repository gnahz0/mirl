#!/bin/bash
# Four-B200 mixed-media memory gate for the production batch shape.
#SBATCH --job-name=mirl_s1_smoke
#SBATCH --partition=b200-batch
#SBATCH --account=ppliang_mit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --exclude=b0028,b0024,b0012
#SBATCH --chdir=/tmp
#SBATCH --output=/work/mit/ppliang_mit/alecz/mirl-qwen35/logs/stage1_smoke_%j.out
#SBATCH --error=/work/mit/ppliang_mit/alecz/mirl-qwen35/logs/stage1_smoke_%j.out

set -eo pipefail

MIRL_CLUSTER_ROOT=/work/mit/ppliang_mit/alecz
cd "$MIRL_CLUSTER_ROOT/mirl-qwen35"
export HF_HOME="$MIRL_CLUSTER_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TMPDIR=/scratch/dvdai_mit/alecz/tmp
export TRITON_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/inductor"
export MIRL_OUTPUT_DIR="/scratch/dvdai_mit/alecz/checkpoints/alignment_smoke_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MIRL_OUTPUT_DIR"

source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate "$MIRL_CLUSTER_ROOT/envs/alec-mv"

CONFIG="$PWD/mirl_ext/alignment/config/stage1_smoke.yaml" \
NUM_GPUS=4 \
bash examples/alignment/run_stage1_qwen35_siglip2.sh
