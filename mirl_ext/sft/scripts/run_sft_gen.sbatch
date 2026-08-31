#!/bin/bash
# Teacher-trace generation on a CPU node.
#   sbatch mirl_ext/sft/run_sft_gen.sbatch
#SBATCH -J sft-gen
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
set -u
export MIRL_OPENAI_KEY="$(<$MIRL_CLUSTER_ROOT/.mirl_openai_key)"
PY="$MIRL_PYENV"/bin/python
DATA=$MIRL_DATA_ROOT
cd $MIRL_CLUSTER_ROOT/mirl-qwen35

$PY mirl_ext/sft/scripts/gen_sft_targets.py --tasks $DATA/split/ts_tasks.staged.jsonl \
    --out $DATA/split/traces_ts.jsonl --concurrency 24 &
$PY mirl_ext/sft/scripts/gen_sft_targets.py --tasks $DATA/split/sft_tasks.staged.jsonl \
    --families climb_train human_behaviour_train tactile_train \
    --out $DATA/split/traces_iv.jsonl --concurrency 24 &
$PY mirl_ext/sft/scripts/gen_sft_targets.py --tasks $DATA/split/hmcq_tasks.staged.jsonl \
    --out $DATA/split/traces_hmcq.jsonl --concurrency 8 &
wait
