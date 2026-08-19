#!/bin/bash
# Launch raw-signal GRPO using the correctly aligned Qwen3-VL vision encoder.
# Submit:   sbatch /work/mit/ppliang_mit/alecz/run_trainedve_raw_b200.sbatch
# Watch:    tail -f /work/mit/ppliang_mit/alecz/logs/trainedve_raw_<jobid>.out
# Cancel:   scancel <jobid>
#
# Config: training script defaults to N_GPUS_PER_NODE=4, ROLLOUT_TP=2 (=> 2 data-parallel
# vLLM replicas), offload OFF, SP=1, gpu_memory_utilization=0.6, all token caps at 15360.
# It respects a Slurm-provided CUDA_VISIBLE_DEVICES, so we request 4 B200s.
# To scale up: set --gpus-per-node=8 here AND export N_GPUS_PER_NODE=8 below.
#SBATCH --job-name=trainedve_raw
#SBATCH --partition=b200-batch
#SBATCH --account=ppliang_mit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclude=b0010   # node hit "Resource temporarily unavailable" during video decode,
                          # crashing vLLM's multimodal placeholder alignment (jobs 116738/117386)
#SBATCH --mem=1600G   # param_offload+optimizer_offload=True parks params/optim in host RAM (~800G+),
                      # plus periodic validation video-decode spike. 1600G matched the config that
                      # survived 14 real training steps + full validation (job 101320).
#SBATCH --time=24:00:00   # partition max (b200-batch TIMELIMIT = 1-00:00:00)
#SBATCH --output=/work/mit/ppliang_mit/alecz/logs/trainedve_raw_%j.out
#SBATCH --error=/work/mit/ppliang_mit/alecz/logs/trainedve_raw_%j.out

set -eo pipefail   # not -u: conda's activate script references unbound vars

# Heavy concurrent video decode (many AgentLoopWorkers, decord+torchvision fallback) can exhaust
# the default open-files/process ulimit on some nodes (seen as "Resource temporarily unavailable"
# during scaling-graph init on b0010, jobs 116738/117386), which corrupts a decoded video and then
# crashes vLLM's multimodal placeholder alignment. Raise the caps defensively.
ulimit -n 65536 2>/dev/null || true
ulimit -u unlimited 2>/dev/null || true

# --- NCCL stability workarounds for intermittent collective hangs on B200 (Blackwell) ---
# Every training run so far has eventually deadlocked inside a torch-NCCL collective
# (ALLGATHER/ALLREDUCE/REDUCE_SCATTER/ALLTOALL, different steps, different nodes), while
# vLLM (custom allreduce kernels, bypasses NCCL) never hangs. Classic Blackwell/NCCL-2.27
# instability. Force NCCL onto its most conservative, best-tested paths:
export NCCL_NVLS_ENABLE=0     # disable NVLink SHARP in-switch reductions (#1 known B200 hang source)
export NCCL_PROTO=Simple      # avoid LL/LL128 low-latency protocols (known hang/corruption issues)
export NCCL_ALGO=Ring         # stick to the most battle-tested algorithm
# Jobs 121287/122477 still hung (same step ~35) WITH the three flags above. NCCL_P2P_DISABLE=1
# was tried in job 122632 but cost ~50% step time (7.4 -> 11 min/step) and was abandoned.
# Instead the env's NCCL was upgraded 2.27.3 -> 2.28.9 (pip nvidia-nccl-cu12), which carries an
# explicit fix: "operation ordering between main thread and proxy thread to prevent hangs".

# NCCL flight recorder: on a collective timeout, dump per-rank traces of the stuck collective
# so we can see exactly which path deadlocked. Dump to /work (node /tmp dies with the job).
# TORCH_FR_BUFFER_SIZE is the current name; keep the deprecated one for compatibility.
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
mkdir -p /work/mit/ppliang_mit/alecz/logs/nccl_traces
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/work/mit/ppliang_mit/alecz/logs/nccl_traces/trace_${SLURM_JOB_ID:-manual}_rank

# --- keep ALL caches off the 93 GB home quota (point everything at /work or /scratch) ---
export HF_HOME=/work/mit/ppliang_mit/alecz/hf_cache
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR=/work/mit/ppliang_mit/alecz/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/work/mit/ppliang_mit/alecz/.cache/inductor
export XDG_CACHE_HOME=/work/mit/ppliang_mit/alecz/.cache/xdg
export VLLM_CACHE_ROOT=/work/mit/ppliang_mit/alecz/.cache/vllm
export RAY_TMPDIR=/scratch/${USER}/alecz/ray_tmp
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$RAY_TMPDIR"

# --- wandb + run identity (script reads PROJECT_NAME/EXPERIMENT_NAME; CKPT_DIR derives from them) ---
export PROJECT_NAME="multiverse"
export WANDB_API_KEY="$(cat /work/mit/ppliang_mit/alecz/.wandb_key)"
export WANDB_ENTITY="aleczhang-massachusetts-institute-of-technology"
export WANDB_RESUME="allow"   # fresh IDs start at 0; later 24h relays can append cleanly
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

# --- activate the Blackwell-built env (prefix env on /work) ---
source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate /work/mit/ppliang_mit/alecz/envs/alec-mv

cd /work/mit/ppliang_mit/alecz/mirl-trainedve

# --- CONTROLLED A/B: baseline (stock Qwen3-VL vision encoder) vs trainedve (our merged VE).
# Both variants train/eval on the SAME parquets + SAME reward code on this cluster; the ONLY
# difference is MODEL_PATH (the vision encoder). Pick via `sbatch --export=ALL,RUN_VARIANT=trainedve`.
# Fresh run ids (…-v2) so relays append cleanly and we never collide with the old merged
# baseline-ve-main history (which caused wandb "step < current step, ignored" drops).
RUN_VARIANT="trainedve"
STOCK_VE=/work/mit/ppliang_mit/alecz/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
ALIGNMENT_BEST=/scratch/dvdai_mit/alecz/checkpoints/alignment_stage1_v2/best/alignment_state.pt
TRAINED_VE=/scratch/dvdai_mit/alecz/models/qwen3vl-8b-trained-ve-correct-v2
case "$RUN_VARIANT" in
  baseline)
    export MODEL_PATH="$STOCK_VE"
    export EXPERIMENT_NAME="baseline-ve"
    export WANDB_RUN_ID="baseline-ve-v4-clean" ;;
  trainedve)
    if [[ ! -f "$TRAINED_VE/CORRECT_STAGE1_VE.json" ]]; then
      echo "[sbatch] exporting correct Stage-1 best VE into full Qwen3-VL checkpoint"
      python /work/mit/ppliang_mit/alecz/mirl-align/scripts/export_stage1_visual_to_hf.py \
        --base-model "$STOCK_VE" \
        --alignment-checkpoint "$ALIGNMENT_BEST" \
        --output-dir "$TRAINED_VE"
    fi
    export MODEL_PATH="$TRAINED_VE"
    export EXPERIMENT_NAME="trainedve-correct-raw"
    export WANDB_RUN_ID="trainedve-correct-raw-v1"
    export DATA_DIR="/work/mit/ppliang_mit/alecz/data/trainedve_raw" ;;
  tstok)
    # A/B for data representation: stock VE, but time-series datasets (ecg/smellnet/haptic_ts)
    # fed as RAW NUMERIC TEXT TOKENS instead of plot images. See data/TIME_SERIES_TOKENS.md.
    export MODEL_PATH="$STOCK_VE"
    export EXPERIMENT_NAME="ts-tokens"
    export WANDB_RUN_ID="ts-tokens-v1"
    export TS_TOKENS=1 ;;
  *) echo "[sbatch] unknown RUN_VARIANT=$RUN_VARIANT (use baseline|trainedve|tstok)"; exit 1 ;;
esac
echo "[sbatch] RUN_VARIANT=$RUN_VARIANT  EXPERIMENT_NAME=$EXPERIMENT_NAME  WANDB_RUN_ID=$WANDB_RUN_ID  TS_TOKENS=${TS_TOKENS:-0}"

echo "==================================================================="
echo "[sbatch] node=$(hostname)  job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] python=$(which python)"
echo "[sbatch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[sbatch] MODEL_PATH=$MODEL_PATH"
nvidia-smi -L || true
echo "==================================================================="

# Hand off to the actual training script. The trained-VE worktree's combined script consumes the
# first token as ENGINE and shifts it, so remaining tokens are forwarded to Hydra as overrides.
# NOTE: the script sets actor_rollout_ref.rollout.enforce_eager=False. If the vLLM rollout
# crashes/hangs during CUDA-graph capture on Blackwell, set line 117 to enforce_eager=True.
#
# mm_processor_cache_gb=0 DISABLES vLLM's multimodal IPC/processor LRU cache. With it enabled
# (default 4GiB), the P0-sender and P1-receiver mirrored LRUs desync under heavy raw-signal +
# CLIMB image churn: the sender strips a tensor believing the receiver still caches that mm_hash,
# the receiver already evicted it, and the engine-core input thread dies with
# "AssertionError: Expected a cached item for mm_hash=..." (job 153477 hung at step 26).
bash examples/multiverse_trainer/combined_qwen3_training.sh vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
