MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RAY_TMPDIR="${RAY_TMPDIR:-${MIRL_ROOT}/.ray_tmp}"
mkdir -p "$RAY_TMPDIR"

: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export QWEN_VL_MAX_IMAGE_TOKENS=1024
export VIDEO_MAX_FRAMES=4
export HYDRA_FULL_ERROR=1

set -x
ENGINE=${1:-vllm}

PROJECT_NAME='multiverse'
EXPERIMENT_NAME='combined_mini_sanity_check'

# Same filtered JSONs as filter_by_token_limit.py (default inputs combined_*_demo_only.json, no --hb-only):
#   data/combined_train_demo_only_filtered_8192.json
#   data/combined_valid_demo_only_filtered_8192.json
TRAIN_FILE="${TRAIN_FILE:-${MIRL_ROOT}/data/combined_train_demo_only_filtered_8192.json}"
VAL_FILE="${VAL_FILE:-${MIRL_ROOT}/data/combined_valid_demo_only_filtered_8192.json}"

mkdir -p "outputs/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=30 \
    data.filter_overlong_prompts_multimodal_num_proc=30 \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    data.max_video_frames=4 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=2048 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="outputs/${EXPERIMENT_NAME}" \
    trainer.val_only=False \
    trainer.test_freq=3 \
    trainer.total_epochs=15 $@