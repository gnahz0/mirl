#!/bin/bash
# Qwen3.5 SFT on two B200s.
#   SMOKE=1 sbatch mirl_ext/sft/run_sft_b200.sbatch
#   sbatch mirl_ext/sft/run_sft_b200.sbatch
#SBATCH -J mirl-sft
#SBATCH -p b200-batch
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --exclude=b0028,b0024,b0012

source $MIRL_CONDA_SH
conda activate $MIRL_PYENV
set -euo pipefail
# Multimedia dataloader workers pass nested tensors to the main process via
# file descriptors (SCM_RIGHTS), so retain a high nofile limit.
ulimit -n 65536 2>/dev/null || true

MIRL_ROOT=$MIRL_CLUSTER_ROOT/mirl-qwen35
PQ=${PQ:-$MIRL_DATA_ROOT/split_grpo/sft_parquet}
MODEL_PATH=${MODEL_PATH:-$MIRL_SCRATCH_ROOT/models/qwen35-stage1-current}
EXPERIMENT_PREFIX=${EXPERIMENT_NAME:-}
NPROC=${NPROC:-2}
RESUME_MODE=${RESUME_MODE:-auto}
if [ "$RESUME_MODE" != "auto" ] && [ "$RESUME_MODE" != "disable" ]; then
    echo "RESUME_MODE must be auto or disable; explicit paths are not audit-bound" >&2
    exit 1
fi
EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
    EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-smoke-hom-${SLURM_JOB_ID:-manual}}
    RESUME_MODE=disable
    EXTRA=(
        data.train_max_samples=256
        data.train_batch_size=8
        trainer.total_epochs=1
        trainer.total_training_steps=4
        "trainer.logger=[console]"
        "checkpoint.save_contents=[model,extra]"
        "checkpoint.load_contents=[model,extra]"
    )
else
    EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-abz-v4}
fi

# Closed-only SFT policy. Keep this explicit so removed families and stale
# open-response parquets in a reused output directory cannot enter training.
CLOSED_SFT_FAMILIES=(
    climb_train
    ecg_train
    human_behaviour_train
    tactile_train
)
TRAIN_PATHS=()
for family in "${CLOSED_SFT_FAMILIES[@]}"; do
    path="$PQ/${family}_sft.parquet"
    if [ ! -f "$path" ]; then
        echo "missing required closed SFT parquet: $path" >&2
        exit 1
    fi
    TRAIN_PATHS+=("$path")
done
TRAIN_FILES="$(IFS=,; echo "${TRAIN_PATHS[*]}")"
for override in "$@"; do
    case "$override" in
        data.train_files=*|+data.train_files=*|++data.train_files=*|~data.train_files|~data.train_files=*|data=*|+data=*|++data=*|~data|~data=*)
            echo "refusing Hydra override that could replace audited data.train_files: $override" >&2
            exit 1
            ;;
    esac
done

cd "$MIRL_ROOT"
AUDIT_SHA=$(python mirl_ext/sft/scripts/verify_sft_artifacts.py \
    --parquet-root "$PQ" \
    --files "${TRAIN_PATHS[@]}" \
    --print-digest)
if [[ ! "$AUDIT_SHA" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid audit digest from verifier: $AUDIT_SHA" >&2
    exit 1
fi
EXPERIMENT_NAME="${EXPERIMENT_PREFIX}-${AUDIT_SHA:0:12}"
echo "verified audited SFT artifacts $AUDIT_SHA; run=$EXPERIMENT_NAME"

# Auto-resume is safe only inside the checkpoint directory bound to this exact
# audit manifest. Any rebuilt parquet/media manifest gets a new directory.
WANDB_KEY_FILE=$MIRL_CLUSTER_ROOT/.wandb_key
if [ ! -r "$WANDB_KEY_FILE" ]; then
    echo "missing W&B key: $WANDB_KEY_FILE" >&2
    exit 1
fi
export WANDB_API_KEY="$(<"$WANDB_KEY_FILE")"
export WANDB_ENTITY=$MIRL_WANDB_ENTITY
export WANDB_MODE=online
export WANDB_RUN_ID="mirl-sft-$EXPERIMENT_NAME"
export WANDB_RESUME=allow
export WANDB_DIR=$MIRL_CLUSTER_ROOT/wandb
export WANDB_DATA_DIR=$MIRL_CLUSTER_ROOT/wandb-data
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR"

# The final save also writes an HF-format model (hf_model) so eval/RL can load
# it directly -- no manual verl.model_merger pass. Resume never loads it.
torchrun --standalone --nproc_per_node=$NPROC -m mirl_ext.sft.sft_trainer \
    model.path=$MODEL_PATH \
    engine.use_torch_compile=False \
    data.custom_cls.path=$MIRL_ROOT/mirl_ext/sft/sft_dataset.py \
    data.custom_cls.name=MIRLSFTDataset \
    data.max_length=24576 \
    data.truncation=error \
    data.max_token_len_per_gpu=24576 \
    data.use_dynamic_bsz=True \
    data.train_batch_size=64 \
    engine.strategy=fsdp2 \
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    "checkpoint.load_contents=[model,optimizer,extra]" \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.lr_scheduler_type=cosine \
    trainer.n_gpus_per_node=$NPROC \
    trainer.total_epochs=2 \
    trainer.save_freq=25 \
    trainer.max_ckpt_to_keep=3 \
    trainer.project_name=mirl-sft \
    "${EXTRA[@]}" \
    "$@" \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=null \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$MIRL_SCRATCH_ROOT/checkpoints/mirl-sft/$EXPERIMENT_NAME \
    data.train_files="[$TRAIN_FILES]"
