#!/bin/bash
# Production Stage-1 Qwen3.5 vision-encoder alignment.
# Submit from the synced cluster checkout (mirl.env supplies account/paths):
#   source mirl.env && sbatch --output "$MIRL_CLUSTER_ROOT/mirl-qwen35/logs/stage1_%j.out" \
#     mirl_ext/alignment/run_stage1_b200.sbatch
#
# The YAML is the experiment definition; this file only supplies cluster paths,
# credentials, caches, and four-GPU launch settings.
#SBATCH --job-name=mirl_align_s1
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=896G
#SBATCH --time=24:00:00
#SBATCH --exclude=b0028,b0024,b0012
#SBATCH --chdir=/tmp

set -eo pipefail
ulimit -n 65536 2>/dev/null || true

MIRL_CLUSTER_ROOT="${MIRL_CLUSTER_ROOT:?source mirl.env before sbatch}"
for attempt in $(seq 1 60); do
  if cd "$MIRL_CLUSTER_ROOT/mirl-qwen35" 2>/dev/null; then
    break
  fi
  if [ "$attempt" -eq 60 ]; then
    echo "FATAL: $MIRL_CLUSTER_ROOT/mirl-qwen35 is unavailable"
    exit 1
  fi
  ls "$(dirname "$MIRL_CLUSTER_ROOT")" >/dev/null 2>&1 || true
  sleep 5
done

export HF_HOME="$MIRL_CLUSTER_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TMPDIR="${MIRL_SCRATCH_ROOT:?}/tmp"
export TRITON_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$MIRL_CLUSTER_ROOT/.cache/inductor"
export XDG_CACHE_HOME="$MIRL_CLUSTER_ROOT/.cache/xdg"
export WANDB_DIR="$MIRL_CLUSTER_ROOT/wandb"
export WANDB_DATA_DIR="$MIRL_CLUSTER_ROOT/wandb-data"
export WANDB_PROJECT=mirl-alignment
export WANDB_ENTITY="${MIRL_WANDB_ENTITY:?}"
export WANDB_RUN_ID="stage1-qwen35-canonical-siglip-${SLURM_JOB_ID}"
export WANDB_RESUME=never
export MIRL_OUTPUT_DIR="${MIRL_SCRATCH_ROOT}/checkpoints/alignment_qwen35_canonical_siglip_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_DATA_DIR" "$MIRL_OUTPUT_DIR"

if [ -r "$MIRL_CLUSTER_ROOT/.wandb_key" ]; then
  export WANDB_API_KEY="$(<"$MIRL_CLUSTER_ROOT/.wandb_key")"
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi

source "${MIRL_CONDA_SH:?}"
conda activate "${MIRL_PYENV:?}"

python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
  echo "FATAL: CUDA is unavailable on $(hostname)"
  exit 1
}

echo "Stage-1 job=$SLURM_JOB_ID node=$(hostname) output=$MIRL_OUTPUT_DIR"
CONFIG="$PWD/mirl_ext/alignment/config/stage1_qwen35_siglip2_aicr.yaml" \
NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}" \
bash examples/alignment/run_stage1_qwen35_siglip2.sh "$@"
