#!/bin/bash
# Production Stage-1 alignment on one MIT ORCD node with two H200s.
# Submit a fresh allocation with:
#   sbatch examples/mirl/slurm/run_stage1_orcd.sbatch
# Or launch it inside an existing allocation with srun.
#SBATCH --job-name=stage1-qwen35-ve
#SBATCH --partition=pi_ppliang
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/alecz/mirl
#SBATCH --output=/orcd/scratch/orcd/010/alecz/mirl-qwen35/logs/stage1_%j.out
#SBATCH --error=/orcd/scratch/orcd/010/alecz/mirl-qwen35/logs/stage1_%j.out

set -eo pipefail
ulimit -n 65536 2>/dev/null || true

MIRL_ROOT=/home/alecz/mirl
ORCD_ROOT=/orcd/scratch/orcd/010/alecz/mirl-qwen35
PYTHON_BIN="$ORCD_ROOT/venv/bin/python"
CONFIG="$MIRL_ROOT/mirl_ext/alignment/config/stage1_qwen35_siglip2_orcd.yaml"
NUM_GPUS="${NUM_GPUS:-2}"
RUN_ID="${MIRL_RUN_ID:-stage1-qwen35-ve-${SLURM_JOB_ID:-manual}}"

cd "$MIRL_ROOT"
export PYTHONPATH="$MIRL_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=torchcodec
export TORCHCODEC_LOG_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="$ORCD_ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ORCD_ROOT/tmp"
export TRITON_CACHE_DIR="$ORCD_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ORCD_ROOT/cache/inductor"
export XDG_CACHE_HOME="$ORCD_ROOT/cache/xdg"
export WANDB_DIR="$ORCD_ROOT/wandb"
export WANDB_PROJECT=mirl-alignment
export WANDB_ENTITY=aleczhang-massachusetts-institute-of-technology
export WANDB_RUN_ID="$RUN_ID"
export WANDB_RESUME=never
export WANDB_MODE="${WANDB_MODE:-online}"
export MIRL_OUTPUT_DIR="$ORCD_ROOT/checkpoints/$RUN_ID"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$WANDB_DIR" "$MIRL_OUTPUT_DIR" "$ORCD_ROOT/logs"

test -x "$PYTHON_BIN"
test -r "$CONFIG"
test -d "$ORCD_ROOT/data"
test -d "$ORCD_ROOT/models/qwen35-ve"
test -d "$ORCD_ROOT/models/siglip2-text"

echo "Stage-1 run=$RUN_ID job=${SLURM_JOB_ID:-none} node=$(hostname) commit=$(git rev-parse HEAD)"
echo "Config=$CONFIG output=$MIRL_OUTPUT_DIR GPUs=$NUM_GPUS"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  -m mirl_ext.alignment.trainer \
  --config "$CONFIG"
