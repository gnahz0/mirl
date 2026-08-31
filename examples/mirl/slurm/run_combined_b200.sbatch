#!/bin/bash
#SBATCH --job-name=mirl-q35-smoke
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=04:00:00

# Default submission is the bounded one-step smoke:
#   sbatch examples/mirl/slurm/run_combined_b200.sbatch
# For a full run, override SMOKE and use the batch partition:
#   sbatch --partition=b200-batch --time=24:00:00 \
#     --export=ALL,SMOKE=0 examples/mirl/slurm/run_combined_b200.sbatch

set -eo pipefail

MIRL_ROOT=$MIRL_CLUSTER_ROOT/mirl-qwen35
MIRL_ENV=$MIRL_PYENV
MIRL_SCRATCH=$MIRL_SCRATCH_ROOT

ulimit -n 65536 2>/dev/null || true
ulimit -u unlimited 2>/dev/null || true

export PATH="${MIRL_ENV}/bin:${PATH}"
export CUDA_HOME="${MIRL_ENV}"
export LD_LIBRARY_PATH="${MIRL_ENV}/targets/x86_64-linux/lib:${MIRL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${MIRL_ENV}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${MIRL_ENV}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${MIRL_ENV}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MIRL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=$MIRL_CLUSTER_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=torchcodec
export TMPDIR="${MIRL_SCRATCH}/tmp-qwen35/${SLURM_JOB_ID}"
export PIP_CACHE_DIR="${MIRL_SCRATCH}/pip-cache-qwen35"
export RAY_TMPDIR=""$MIRL_SCRATCH_ROOT"/ray_tmp/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${MIRL_SCRATCH}/cache-qwen35/xdg"
export TRITON_CACHE_DIR="${MIRL_SCRATCH}/cache-qwen35/triton"
export TORCHINDUCTOR_CACHE_DIR="${MIRL_SCRATCH}/cache-qwen35/inductor"
export VLLM_CACHE_ROOT="${MIRL_SCRATCH}/cache-qwen35/vllm"
export FLASHINFER_WORKSPACE_BASE="${MIRL_SCRATCH}/cache-qwen35/flashinfer"
export NCCL_NVLS_ENABLE=0
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$MIRL_CLUSTER_ROOT/logs/qwen35_nccl_${SLURM_JOB_ID}_rank"

export SMOKE="${SMOKE:-1}"
export N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export FSDP_SIZE="${FSDP_SIZE:-2}"
export MODEL_PATH="${MODEL_PATH:-$MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
if [[ "${SMOKE}" == "0" && -f $MIRL_CLUSTER_ROOT/.wandb_key ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(<$MIRL_CLUSTER_ROOT/.wandb_key)"
fi

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${RAY_TMPDIR}" "${FLASHINFER_WORKSPACE_BASE}" $MIRL_CLUSTER_ROOT/logs
cd "${MIRL_ROOT}"

echo "job=${SLURM_JOB_ID} node=$(hostname) smoke=${SMOKE}"
echo "python=$(command -v python) model=Qwen3.5-9B"
nvidia-smi -L

python environments/mirl-qwen35/verify_environment.py \
  --cuda --model-snapshot "${MODEL_PATH}"

exec bash examples/mirl/multiverse/run_qwen35_grpo.sh
