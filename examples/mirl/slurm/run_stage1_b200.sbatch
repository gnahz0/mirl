#!/bin/bash
# Production Stage-1 Qwen3.5 vision-encoder alignment.
# Submit from the synced cluster checkout:
#   sbatch examples/mirl/slurm/run_stage1_b200.sbatch
#
# The YAML is the experiment definition; this file only supplies cluster paths,
# credentials, caches, and four-GPU launch settings.
#SBATCH --job-name=tactile_4tok
#SBATCH --partition=b200-batch
#SBATCH --account=ppliang_mit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=896G
#SBATCH --time=24:00:00
#SBATCH --exclude=b0028,b0024,b0012
#SBATCH --chdir=/tmp
#SBATCH --output=/work/mit/ppliang_mit/alecz/mirl-qwen35-tactile/logs/tactile_%j.out
#SBATCH --error=/work/mit/ppliang_mit/alecz/mirl-qwen35-tactile/logs/tactile_%j.out

set -eo pipefail
ulimit -n 65536 2>/dev/null || true

MIRL_CLUSTER_ROOT=/work/mit/ppliang_mit/alecz
cd "$MIRL_CLUSTER_ROOT/mirl-qwen35-tactile"

export HF_HOME="$MIRL_CLUSTER_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TMPDIR=/scratch/dvdai_mit/alecz/tmp
export TRITON_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/inductor"
export XDG_CACHE_HOME="$MIRL_CLUSTER_ROOT/.cache/xdg"
export WANDB_DIR=/scratch/dvdai_mit/alecz/wandb
export WANDB_PROJECT=mirl-alignment
export WANDB_ENTITY=aleczhang-massachusetts-institute-of-technology
export WANDB_RUN_ID="tactile-4token-${SLURM_JOB_ID}"
export WANDB_RESUME=never
export MIRL_OUTPUT_DIR="/scratch/dvdai_mit/alecz/checkpoints/tactile_4token_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$WANDB_DIR" "$MIRL_OUTPUT_DIR"

if [ -r "$MIRL_CLUSTER_ROOT/.wandb_key" ]; then
  export WANDB_API_KEY="$(<"$MIRL_CLUSTER_ROOT/.wandb_key")"
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi

source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate "$MIRL_CLUSTER_ROOT/envs/alec-mv"

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
  echo "FATAL: CUDA is unavailable on $(hostname)"
  exit 1
}

echo "Stage-1 job=$SLURM_JOB_ID node=$(hostname) output=$MIRL_OUTPUT_DIR"
CONFIG="$PWD/mirl_ext/alignment/config/stage1_qwen35_siglip2.yaml" \
NUM_GPUS=2 \
bash examples/alignment/run_stage1_qwen35_siglip2.sh "$@"
