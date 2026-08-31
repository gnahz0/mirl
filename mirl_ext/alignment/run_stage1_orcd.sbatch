#!/bin/bash
# Production Stage-1 alignment on one MIT ORCD node with two H200s.
# Submit a fresh allocation with (your env file supplies cluster specifics):
#   source mirl.env && sbatch --partition "$MIRL_ORCD_PARTITION" --account "$MIRL_ORCD_ACCOUNT" \
#     --chdir "$MIRL_ORCD_REPO" --output "$MIRL_ORCD_ROOT/logs/stage1_%j.out" \
#     mirl_ext/alignment/run_stage1_orcd.sbatch
# Or launch it inside an existing allocation with srun.
# OmegaConf overrides may be appended, for example:
#   bash "$0" train.num_train_epochs=2 train.lr=3e-6
#SBATCH --job-name=stage1-qwen35-ve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00

set -eo pipefail
ulimit -n 65536 2>/dev/null || true

MIRL_ROOT="${MIRL_ORCD_REPO:-$SLURM_SUBMIT_DIR}"
ORCD_ROOT="${MIRL_ORCD_ROOT:?source mirl.env before sbatch}"
PYTHON_BIN="$ORCD_ROOT/venv/bin/python"
CONFIG="$MIRL_ROOT/mirl_ext/alignment/config/stage1_qwen35_siglip2_orcd.yaml"
NUM_GPUS="${NUM_GPUS:-2}"
RUN_ID="${MIRL_RUN_ID:-stage1-qwen35-ve-${SLURM_JOB_ID:-manual}}"

cd "$MIRL_ROOT"
export PYTHONPATH="$MIRL_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
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
export WANDB_ENTITY="${MIRL_WANDB_ENTITY:?}"
export WANDB_RUN_ID="$RUN_ID"
export WANDB_RESUME=never
export WANDB_MODE="${WANDB_MODE:-online}"
export MIRL_OUTPUT_DIR="$ORCD_ROOT/checkpoints/$RUN_ID"
export MIRL_ALIGN_DATA_ROOT="$ORCD_ROOT/data"
export MIRL_QWEN35_PATH="$ORCD_ROOT/models/qwen35-ve"
export MIRL_SIGLIP2_PATH="$ORCD_ROOT/models/siglip2-text"

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
  -m mirl_ext.alignment.train \
  --config "$CONFIG" \
  "$@"
