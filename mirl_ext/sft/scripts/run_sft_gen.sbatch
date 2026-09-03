#!/bin/bash
# Teacher-trace generation on a CPU node (internet-capable). Parameterized:
#   TASKS=$DATA/split/<tasks>.staged.jsonl OUT=$DATA/split/<traces>.jsonl \
#     sbatch --export=ALL mirl_ext/sft/scripts/run_sft_gen.sbatch
# Optional: FAMILIES="climb_train tactile_train", CONCURRENCY=8, EXTRA_ARGS=...
# The v1 corpus (traces_50.jsonl) is complete; this runner exists for the next
# corpus. Current tactile traces use full synchronized RGB+heatmap composites.
#SBATCH -J sft-gen
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
SUBMIT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -r "$SUBMIT_ROOT/mirl.env" ]; then
    source "$SUBMIT_ROOT/mirl.env"
fi
set -u
: "${MIRL_CLUSTER_ROOT:?source mirl.env or export MIRL_CLUSTER_ROOT before submission}"
: "${MIRL_PYENV:?source mirl.env or export MIRL_PYENV before submission}"
KEY_FILE="${MIRL_OPENAI_KEY_FILE:-$MIRL_CLUSTER_ROOT/.config/mirl/microsoft_openai_key}"
if [ -z "${MIRL_OPENAI_KEY:-}" ]; then
    if [ ! -r "$KEY_FILE" ]; then
        echo "missing teacher API key: $KEY_FILE" >&2
        exit 1
    fi
    export MIRL_OPENAI_KEY="$(<"$KEY_FILE")"
fi
PY="$MIRL_PYENV"/bin/python
cd $MIRL_CLUSTER_ROOT/mirl-qwen35

$PY mirl_ext/sft/scripts/gen_sft_targets.py --tasks "$TASKS" --out "$OUT" \
    ${FAMILIES:+--families $FAMILIES} \
    --concurrency "${CONCURRENCY:-8}" ${EXTRA_ARGS:-}
