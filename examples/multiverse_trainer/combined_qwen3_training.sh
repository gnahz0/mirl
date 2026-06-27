MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# verl is the local repo (not pip-installed), so run from the repo root and put it
# on PYTHONPATH; otherwise `python -m verl.trainer.main_ppo` fails with ModuleNotFoundError.
cd "$MIRL_ROOT" || exit 1
export PYTHONPATH="${MIRL_ROOT}:${PYTHONPATH}"
# Ray plasma sockets are AF_UNIX; full path must be <= 107 bytes — use a short scratch dir, not repo/.ray_tmp.
export RAY_TMPDIR="${RAY_TMPDIR:-/scratch/${USER}/ray_tmp}"
mkdir -p "$RAY_TMPDIR"

# Multimodal token control is done via *config* knobs (data.max_video_frames /
# data.max_image_tokens below), NOT env vars: the rollout/agent-loop runs in Ray workers that
# don't inherit the driver shell env, and RLHFDataset.process_vision_info reads these from config.
# - Video: data.max_video_frames caps frames (qwen_vl_utils samples down to it).
# - Image: data.max_image_tokens caps resolution. Rollout images bypass verl's process_image and
#   go straight through qwen_vl_utils, whose default is ~16384 visual tokens -- a high-res CLIMB
#   image then overflows max_model_len ("Prompt length (16337) exceeds ..."). Units are patch-14
#   tokens; model tokens ~= value / 4 (12288 -> ~3072 tokens, ~1550x1550 px). The cap is adaptive:
#   per-image ceiling = max_image_tokens, but multi-image prompts (CLIMB has up to 4 imgs/sample)
#   bound each image by max_image_tokens_total / n_images so the combined visual tokens still fit.
#
# Prompt budget: the agent-loop pads each rollout prompt to rollout.prompt_length but does NOT
# truncate, so any prompt longer than it crashes the batch torch.cat ("Sizes of tensors must
# match ... Expected 8192 but got 8319"). human_behaviour transcripts reach ~7527 text tokens and
# also carry a video (~2942 tokens) -> up to ~10.5k. So max_prompt_length=11264 (+4096 response =
# max_model_len 15360) to fit the long-text tail without dropping data. All other sources have
# tiny text (<=309 tokens), so their image/video caps already keep them well under budget.

# Respect an externally-provided CUDA_VISIBLE_DEVICES (e.g. from wait_for_gpus_and_run.sh);
# fall back to a fixed set when launched directly.
: "${CUDA_VISIBLE_DEVICES:=1,2,4,5}"
export CUDA_VISIBLE_DEVICES
set -x
ENGINE=${1:-vllm}

# Combined training: HB + CLIMB on Qwen3-VL-8B with GRPO
# Uses default_compute_score dispatch (no custom_reward_function needed)
# Data is in the new unified format with data_source + prompt + reward_model
#
# Data: per-dataset parquet splits in data/ (smellnet, ecg, haptic_ts, climb,
# human_behaviour, tactile) x {train,valid}. All six per split are passed as a
# list to data.train_files/data.val_files and concatenated by RLHFDataset.
# Time-series datasets (smellnet/ecg/haptic_ts) are rendered to images; no raw time-series.
# Override TRAIN_FILE/VAL_FILE (single path or Hydra list literal) to use other data.
#
# Checkpoints saved to: checkpoints/multiverse/combined_hb_climb_qwen3_grpo/
# Validation outputs:   outputs/combined_hb_climb_qwen3_grpo/

PROJECT_NAME='multiverse'
EXPERIMENT_NAME='combined_hb_climb_qwen3_grpo'

# Per-dataset splits in data/ (6 datasets x {train,valid}).
# RLHFDataset accepts a list of files and concatenates them, so pass all six per split.
# Files are parquet with a unified schema (images=List{image}, videos=List{video,min_frames,max_frames},
# extra_info=string). parquet carries its schema in metadata, so concatenate_datasets aligns cleanly
# across datasets that have empty image/video lists. Re-generate via scripts/build_parquet.py if JSONs change.
DATA_DIR="${MIRL_ROOT}/data"
_TRAIN_NAMES=(smellnet_train ecg_train haptic_ts_train climb_train human_behaviour_train tactile_train)
# Validation is subsampled to fit a full pass in ~1h (was ~4h). The cost is VIDEO decode, so only
# the video-heavy splits are capped: *_valid_fast are stratified per-data_source caps (every task
# metric still has samples), built by scripts/subsample_valid_fast.py: tactile 9407->895,
# human_behaviour 2000->351. ecg/smellnet/haptic/climb are kept full (images are cheap to decode).
# Swap back to the un-suffixed splits (e.g. tactile_valid) for a full-fidelity eval.
_VALID_NAMES=(smellnet_valid ecg_valid haptic_ts_valid climb_valid human_behaviour_valid_fast tactile_valid_fast)

# Build a Hydra list literal: ["<dir>/a.parquet","<dir>/b.parquet",...]
_join_files() {
    local dir="$1"; shift
    local out=""
    for n in "$@"; do
        [ -n "$out" ] && out="${out},"
        out="${out}\"${dir}/${n}.parquet\""
    done
    printf '[%s]' "$out"
}

TRAIN_FILE="${TRAIN_FILE:-$(_join_files "$DATA_DIR" "${_TRAIN_NAMES[@]}")}"
VAL_FILE="${VAL_FILE:-$(_join_files "$DATA_DIR" "${_VALID_NAMES[@]}")}"

mkdir -p "outputs/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=11264 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    data.max_video_frames=8 \
    +data.max_video_bytes=629145600 \
    +data.max_image_tokens=12288 \
    +data.max_image_tokens_total=24576 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=15360 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=15360 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=15360 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15360 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=15360 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=4096 \
    +ray_init.num_cpus=16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="outputs/${EXPERIMENT_NAME}" \
    trainer.val_only=False \
    trainer.test_freq=15 \
    trainer.total_epochs=10 $@
