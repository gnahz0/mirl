#!/bin/bash
# Compatibility entrypoint. The canonical SFT launcher is in mirl_ext/sft/.

#SBATCH -J mirl-sft
#SBATCH -p b200-batch
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=b0028,b0024,b0012
#SBATCH -o /work/mit/ppliang_mit/alecz/logs/mirl-sft-%j.out

exec "$(cd "$(dirname "$0")/../../../mirl_ext/sft" && pwd)/run_sft_b200.sbatch"
