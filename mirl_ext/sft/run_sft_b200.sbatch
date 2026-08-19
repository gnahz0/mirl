#!/bin/bash
# Qwen3.5 SFT on two B200s.
#   SMOKE=1 sbatch mirl_ext/sft/run_sft_b200.sbatch
#   sbatch mirl_ext/sft/run_sft_b200.sbatch
#SBATCH -J mirl-sft
#SBATCH -p b200-batch
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=b0028,b0024,b0012
#SBATCH -o /work/mit/ppliang_mit/alecz/logs/mirl-sft-%j.out

source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate /work/mit/ppliang_mit/alecz/envs/alec-mv
set -euo pipefail

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

MIRL_ROOT=/work/mit/ppliang_mit/alecz/mirl-qwen35
PQ=/work/mit/ppliang_mit/alecz/data/split_grpo/sft_parquet
MODEL_PATH=${MODEL_PATH:-/scratch/dvdai_mit/alecz/models/qwen35-stage1-current}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-abz-v2.1}
NPROC=2
EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
    EXPERIMENT_NAME=smoke
    EXTRA=(data.train_max_samples=64 trainer.total_epochs=1 "trainer.logger=[console]")
fi

WANDB_KEY_FILE=/work/mit/ppliang_mit/alecz/.wandb_key
if [ ! -r "$WANDB_KEY_FILE" ]; then
    echo "missing W&B key: $WANDB_KEY_FILE" >&2
    exit 1
fi
export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
export WANDB_ENTITY=aleczhang-massachusetts-institute-of-technology
export WANDB_MODE=online
export WANDB_RUN_ID="mirl-sft-$EXPERIMENT_NAME"
export WANDB_RESUME=allow
export WANDB_DIR=/work/mit/ppliang_mit/alecz/wandb
export WANDB_DATA_DIR=/work/mit/ppliang_mit/alecz/wandb-data
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR"

TRAIN_FILES="[\
$PQ/climb_train_sft.parquet,\
$PQ/human_behaviour_train_sft.parquet,\
$PQ/tactile_train_sft.parquet,\
$PQ/ecg_train_sft.parquet,\
$PQ/haptic_mcq_train_sft.parquet,\
$PQ/human_behaviour_train_open_sft.parquet,\
$PQ/tactile_train_open_sft.parquet,\
$PQ/haptic_ts_train_open_sft.parquet]"

cd "$MIRL_ROOT"
torchrun --standalone --nproc_per_node=$NPROC -m verl.trainer.sft_trainer \
    model.path=$MODEL_PATH \
    engine.use_torch_compile=False \
    data.custom_cls.path=$MIRL_ROOT/mirl_ext/sft/sft_dataset.py \
    data.custom_cls.name=MIRLSFTDataset \
    data.train_files="$TRAIN_FILES" \
    data.max_length=24576 \
    data.truncation=error \
    data.max_token_len_per_gpu=24576 \
    data.use_dynamic_bsz=False \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.lr_scheduler_type=cosine \
    trainer.n_gpus_per_node=$NPROC \
    trainer.total_epochs=2 \
    trainer.save_freq=25 \
    trainer.max_ckpt_to_keep=3 \
    trainer.project_name=mirl-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=/scratch/dvdai_mit/alecz/checkpoints/mirl-sft/$EXPERIMENT_NAME \
    "${EXTRA[@]}"
