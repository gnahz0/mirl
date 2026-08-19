#!/bin/bash
# Compatibility entrypoint. The canonical SFT launcher is in mirl_ext/sft/.

#SBATCH -J sft-gen
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /work/mit/ppliang_mit/alecz/logs/sft-gen-%j.out

exec "$(cd "$(dirname "$0")/../../../mirl_ext/sft" && pwd)/run_sft_gen.sbatch"
