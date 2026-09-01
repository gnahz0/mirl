#!/usr/bin/env bash
# Qwen3.5-9B GRPO over MIRL's mixed image/video/time-series datasets.
set -euo pipefail

MIRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MIRL_ENV_PREFIX="${MIRL_ENV_PREFIX:-$MIRL_PYENV}"
PYTHON="${PYTHON:-${MIRL_ENV_PREFIX}/bin/python}"
DATA_ROOT="${DATA_ROOT:-$MIRL_DATA_ROOT}"
MODEL_PATH="${MODEL_PATH:-$MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
SMOKE="${SMOKE:-0}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-${N_GPUS_PER_NODE}}"
FSDP_SIZE="${FSDP_SIZE:-${N_GPUS_PER_NODE}}"
# Qwen3.5-9B interleaves Gated Delta Net and full-attention layers. Keep
# padding removal and Ulysses SP disabled until upstream adds packed-sequence
# support for the GDN path.
SP_SIZE="${SP_SIZE:-1}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${MIRL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-$MIRL_CLUSTER_ROOT/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-torchcodec}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-50}"
export TMPDIR="${TMPDIR:-$MIRL_SCRATCH_ROOT/tmp-qwen35/${SLURM_JOB_ID:-manual}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MIRL_SCRATCH_ROOT/pip-cache-qwen35}"
export RAY_TMPDIR="${RAY_TMPDIR:-$MIRL_SCRATCH_ROOT/ray_tmp/${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MIRL_SCRATCH_ROOT/cache-qwen35/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MIRL_SCRATCH_ROOT/cache-qwen35/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MIRL_SCRATCH_ROOT/cache-qwen35/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MIRL_SCRATCH_ROOT/cache-qwen35/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$MIRL_SCRATCH_ROOT/cache-qwen35/flashinfer}"
export CUDA_HOME="${CUDA_HOME:-${MIRL_ENV_PREFIX}}"
export CPATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${MIRL_ENV_PREFIX}/targets/x86_64-linux/lib:${MIRL_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${RAY_TMPDIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_BASE}"

join_files() {
    local suffix="$1"
    shift
    local output="" name
    for name in "$@"; do
        [[ -n "${output}" ]] && output+=","
        output+="\"${DATA_ROOT}/${name}${suffix}.parquet\""
    done
    printf '[%s]' "${output}"
}

# Training reads the RL HALF of the 50:50 group split (split_sft_rl.py); the
# root-level <fam>_train.parquet files are the UNSPLIT corpora and overlap the
# SFT half. Validation files live at the root and were never split.
TRAIN_ROOT="${TRAIN_ROOT:-${DATA_ROOT}/split_grpo/rl}"
join_train() {
    local suffix="$1"
    shift
    local output="" name
    for name in "$@"; do
        [[ -n "${output}" ]] && output+=","
        output+="\"${TRAIN_ROOT}/${name}${suffix}.parquet\""
    done
    printf '[%s]' "${output}"
}

if [[ "${TS_NATIVE:-0}" == "1" ]]; then
    # Stage-1 pseudo-video strips for the ts families (mirl_ext/rl/ts_native_DESIGN.md).
    # haptic_ts is 100% open free-text (haptic_tactile) — ungradable, excluded from RL.
    train_files="$(join_train _tsnative ecg_train),$(join_train '' climb_train human_behaviour_train_closed tactile_train_closed)"
    train_files="${train_files//],[/,}"  # merge into ONE list: `[a],[b]` is a Hydra choice sweep and errors in run mode
    val_files="$(join_files _tsnative ecg_valid),$(join_files '' climb_valid human_behaviour_valid_fast_closed tactile_valid_fast_closed)"
    val_files="${val_files//],[/,}"
elif [[ "${TS_TOKENS:-0}" == "1" ]]; then
    # Historical raw-numeric-text A/B; its _tstok parquets predate the RL-half
    # split and no builder exists in this repo (mirl_ext/rl/ts_native_DESIGN.md).
    echo "TS_TOKENS=1 is historical (no _tstok parquets in split_grpo/rl); use TS_NATIVE=1" >&2
    exit 2
else
    # haptic_ts excluded: 100% open free-text, ungradable for RL; closed variants
    # strip the open sources from tactile/human_behaviour (28% of each).
    train_files="$(join_train '' ecg_train climb_train human_behaviour_train_closed tactile_train_closed)"
    val_files="$(join_files '' ecg_valid climb_valid human_behaviour_valid_fast_closed tactile_valid_fast_closed)"
fi

PROJECT_NAME="${PROJECT_NAME:-multiverse-qwen35}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-combined-qwen35-9b}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-11264}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-15360}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_N="${ROLLOUT_N:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
LOGGER="${LOGGER:-['console','wandb']}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:-30}"
TEST_FREQ="${TEST_FREQ:-30}"
OFFLOAD_POLICY="${OFFLOAD_POLICY:-True}"
FILTER_WORKERS="${FILTER_WORKERS:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"

if [[ "${SMOKE}" == "1" ]]; then
    smoke_file="${DATA_ROOT}/qwen35_smoke.parquet"
    "${PYTHON}" -m mirl_ext.data.build_smoke --data-root "${DATA_ROOT}" --output "${smoke_file}"
    train_files="[\"${smoke_file}\"]"
    val_files="[\"${smoke_file}\"]"
    EXPERIMENT_NAME="${SMOKE_EXPERIMENT_NAME:-smoke-qwen35-9b}"
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH_SIZE=8
    MAX_PROMPT_LENGTH=4096
    MAX_RESPONSE_LENGTH=128
    MAX_MODEL_LEN=4224
    MAX_BATCHED_TOKENS=4096
    ROLLOUT_N=2
    GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.4}"
    TOTAL_TRAINING_STEPS=1
    LOGGER="['console']"
    VAL_BEFORE_TRAIN=False
    SAVE_FREQ=-1
    TEST_FREQ=-1
    OFFLOAD_POLICY=False
    FILTER_WORKERS=null
    DATALOADER_WORKERS=0
    SP_SIZE="${SMOKE_SP_SIZE:-1}"
fi

# Force OUR wandb identity: the shared account's ~/.netrc holds a colleague's
# login, and Ray workers resolved to it once (run landed in the wrong entity).
# Env key + entity beat the netrc (verified); fail loudly if the key is absent.
if [[ "${SMOKE}" != "1" ]]; then
    WANDB_KEY_FILE="$MIRL_CLUSTER_ROOT/.wandb_key"
    if [[ ! -r "${WANDB_KEY_FILE}" ]]; then
        echo "missing W&B key: ${WANDB_KEY_FILE}" >&2
        exit 1
    fi
    export WANDB_API_KEY="$(<"${WANDB_KEY_FILE}")"
    export WANDB_ENTITY="${MIRL_WANDB_ENTITY:?source mirl.env first}"
    # Shared-account netrc trap, layer 2: some Ray worker-spawn path resolved
    # the COLLEAGUE's key from /home/dvdai_mit despite driver env forcing
    # (jobs 632349/634092: actor "logged in as weianxie" -> CommError). Point
    # HOME at our namespace so ~/.netrc is OUR netrc in every child process.
    export HOME="$MIRL_CLUSTER_ROOT"
    echo "wandb: forcing entity=${WANDB_ENTITY} (key file: ${WANDB_KEY_FILE}; HOME=${HOME})"
fi

CKPT_DIR="${CKPT_DIR:-$MIRL_SCRATCH_ROOT/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-$MIRL_CLUSTER_ROOT/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

args=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # Ray dashboard is unused overhead and its MetricsHead subprocess
    # intermittently fails to start on shared nodes, killing the whole run
    # (job 630990: "EOF from pipe" after 1h18m of startup retries).
    +ray_kwargs.ray_init.include_dashboard=False
    "data.train_files=${train_files}"
    "data.val_files=${val_files}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    data.val_batch_size=64
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
    "data.dataloader_num_workers=${DATALOADER_WORKERS}"
    data.truncation=error
    data.image_patch_size=16
    data.image_key=images
    data.video_key=videos
    "data.custom_cls.path=${MIRL_ROOT}/mirl_ext/data/dataset.py"
    data.custom_cls.name=MIRLDataset
    +data.max_video_frames=8
    +data.max_video_bytes=52428800
    +data.max_image_tokens=12288
    +data.max_image_tokens_total=24576
    "reward.custom_reward_function.path=${MIRL_ROOT}/mirl_ext/rewards/combined.py"
    reward.custom_reward_function.name=compute_score
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}"
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${OFFLOAD_POLICY}"
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}"
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    trainer.critic_warmup=0
    "trainer.logger=${LOGGER}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    trainer.balance_batch=False
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    trainer.total_epochs=10
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
)

cd "${MIRL_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exec "${PYTHON}" -m verl.trainer.main_ppo "${args[@]}" --cfg job "$@"
fi
exec "${PYTHON}" -m verl.trainer.main_ppo "${args[@]}" "$@"
