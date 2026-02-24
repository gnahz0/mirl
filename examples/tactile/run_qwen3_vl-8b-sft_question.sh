#!/usr/bin/env bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
backend=${BACKEND:-megatron}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

DATA_DIR="$HOME/scratch/raofu/3DHaptic"
train_path="${DATA_DIR}/annotation_verl_reasoning_sft_split_question_train.parquet"
test_path="${DATA_DIR}/annotation_verl_reasoning_sft_split_question_test_mini.parquet"

MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True"

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS:-4} \
    ${ENTRYPOINT} \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=32 \
    data.max_length=16384 \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=16384 \
    data.ignore_input_ids_mismatch=True \
    model.path=Qwen/Qwen3-VL-8B-Instruct \
    model.use_remove_padding=True \
    ${MEGATRON_ENGINE_CONFIG} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tactile-sft' \
    trainer.experiment_name='qwen3vl_sft_split_question_official' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.resume_mode=auto \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents=[model,optimizer,extra] $@
