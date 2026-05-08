MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Ray plasma uses AF_UNIX; keep socket path short.
export RAY_TMPDIR="${RAY_TMPDIR:-/scratch/${USER}/ray_tmp}"
mkdir -p "$RAY_TMPDIR"

: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HYDRA_FULL_ERROR=1
export NCCL_RAS_ENABLE=0
export VLLM_USE_V1=1
export QWEN_VL_MAX_IMAGE_TOKENS=1024
export RAY_SYSTEM_MEMORY=$((450 * 1024 * 1024 * 1024))


set -x
ENGINE=${1:-vllm}

PROJECT_NAME='multiverse'
EXPERIMENT_NAME='combined_full_run'

TRAIN_FILE="${TRAIN_FILE:-${MIRL_ROOT}/data/combined_train_full.json}"
VAL_FILE="${VAL_FILE:-${MIRL_ROOT}/data/combined_valid_mini.json}"

mkdir -p "outputs/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    data.max_video_frames=6 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/scratch/alecz/checkpoints/multiverse/combined_full_run \
    trainer.save_freq=30 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.validation_data_dir="outputs/${EXPERIMENT_NAME}" \
    trainer.val_only=False \
    trainer.test_freq=20 \
    +data.seed=42 \
    data.val_max_samples=1000 \
    +data.balanced_sampling_key='data_source' \
    trainer.total_epochs=3 $@
