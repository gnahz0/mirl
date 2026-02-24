#!/usr/bin/env bash
set -x

# Merge Megatron SFT checkpoints to HuggingFace format for all 4 splits (skip participant).

PROJECT_NAME="tactile-sft"

for SPLIT_NAME in date glove question task; do
    echo "============================================"
    echo "Merging SFT checkpoint for split: ${SPLIT_NAME}"
    echo "============================================"

    EXPERIMENT_NAME="qwen3vl_sft_split_${SPLIT_NAME}_official"
    CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    OUTPUT_DIR="./outputs/${EXPERIMENT_NAME}"

    LATEST_STEP=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST_STEP" ]; then
        echo "WARNING: No checkpoint found for split '${SPLIT_NAME}' in ${CKPT_DIR}, skipping."
        continue
    fi
    CKPT_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}"
    echo "Found checkpoint at: ${CKPT_PATH}"

    mkdir -p "${OUTPUT_DIR}"

    python -m verl.model_merger merge \
        --backend megatron \
        --local_dir "${CKPT_PATH}" \
        --target_dir "${OUTPUT_DIR}"

    echo "SFT model for split '${SPLIT_NAME}' saved to: ${OUTPUT_DIR}"
done

echo ""
echo "All merges done!"
