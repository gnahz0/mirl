#!/bin/bash
# Launch Stage-1 multimodal alignment (VE retrain) on B200 via Slurm.
# Trains the exact Qwen3.5 vision encoder on the verified raw-signal data:
#   ts branch  = SmellNet/ECG direct-value pseudo-images + haptic pseudo-video
#   img branch = climb + human_behaviour + tactile (distillation vs frozen VE)
# Indexes live under /work/.../data/trainedve_raw; media paths point to /scratch.
#
# Submit:  sbatch /work/mit/ppliang_mit/alecz/mirl-qwen35/examples/mirl/slurm/run_stage1_b200.sbatch
# Watch:   tail -f /work/mit/ppliang_mit/alecz/logs/align_stage1_<jobid>.out
#SBATCH --job-name=mirl_align_s1
#SBATCH --partition=b200-batch
#SBATCH --account=ppliang_mit
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/work/mit/ppliang_mit/alecz/logs/align_stage1_%j.out
#SBATCH --error=/work/mit/ppliang_mit/alecz/logs/align_stage1_%j.out

set -eo pipefail
ulimit -n 65536 2>/dev/null || true

# Guard against transient /work automount glitches at job start (job 131373 died in 1s
# with "Too many levels of symbolic links" chdir'ing to /work). --chdir=/tmp keeps Slurm
# from failing on the workdir; here we wait for the /work automount to resolve cleanly.
ROOT=/work/mit/ppliang_mit/alecz
for i in $(seq 1 60); do
  if cd "$ROOT" 2>/dev/null && [ -d "$ROOT/mirl-qwen35" ]; then echo "[sbatch] /work ready (attempt $i)"; break; fi
  echo "[sbatch] /work not ready yet (attempt $i) — nudging automount, sleeping 5s"; ls /work/mit/ppliang_mit >/dev/null 2>&1 || true; sleep 5
done
cd "$ROOT" || { echo "[sbatch] FATAL: $ROOT unreachable after retries"; exit 1; }

# --- caches off the home quota ---
export HF_HOME=/work/mit/ppliang_mit/alecz/hf_cache
export HF_HUB_OFFLINE=0
export TRITON_CACHE_DIR=/work/mit/ppliang_mit/alecz/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/work/mit/ppliang_mit/alecz/.cache/inductor
export XDG_CACHE_HOME=/work/mit/ppliang_mit/alecz/.cache/xdg
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# --- wandb ---
export WANDB_PROJECT="mirl-alignment"
export WANDB_API_KEY="$(cat /work/mit/ppliang_mit/alecz/.wandb_key)"
export WANDB_ENTITY="aleczhang-massachusetts-institute-of-technology"
# New lineage for the unified SmellNet/ECG raster + NaN-filtered ECG population.
# Do not append these metrics to an older pseudo-video run.
export WANDB_RUN_ID="stage1-qwen35-siglip2-standard-hparams-v2"
export WANDB_RESUME="allow"

# --- env ---
source /home/dvdai_mit/miniconda3/etc/profile.d/conda.sh
conda activate /work/mit/ppliang_mit/alecz/envs/alec-mv

# Qwen3.5 worktree with the ported alignment package.
cd /work/mit/ppliang_mit/alecz/mirl-qwen35

# 4-GPU DDP. YAML has 7,914 single-GPU optimizer steps for ~3 epochs over 168,848
# rows at batch 64. Divide steps/warmup by four to preserve the sample budget.
export NUM_GPUS=4
CKPT=/scratch/dvdai_mit/alecz/checkpoints/alignment_qwen35_siglip2_standard_hparams_v2
mkdir -p "$CKPT"

echo "==================================================================="
echo "[sbatch] node=$(hostname) job=${SLURM_JOB_ID:-NA} NUM_GPUS=$NUM_GPUS"
echo "[sbatch] python=$(which python)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
echo "==================================================================="

bash examples/alignment/run_stage1_qwen35_siglip2.sh \
  train.total_steps=1977 \
  train.warmup_steps=100 \
  train.ckpt_every=250 \
  train.output_dir="$CKPT"
