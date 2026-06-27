#!/usr/bin/env bash
# Wait until NEED_GPUS GPUs are completely free (no compute processes and low memory use),
# then pin CUDA_VISIBLE_DEVICES to those GPUs and launch the combined Qwen3-VL training script.
#
# Usage (from anywhere):
#   bash examples/multiverse_trainer/wait_for_gpus_and_run.sh
# Tunables (env vars):
#   NEED_GPUS=4            how many free GPUs are required
#   MEM_USED_THRESH_MIB=1000   a GPU counts as free only if used mem < this
#   POLL_SECS=30          seconds between checks while waiting
#   STABLE_CHECKS=2       require the same GPUs free this many consecutive checks (anti-race)
#   STABLE_GAP_SECS=10    seconds between the stability re-checks
#   TRAIN_SCRIPT=...      path to the training script (defaults to the sibling script)
# Any extra args are forwarded to the training script.
set -uo pipefail

NEED_GPUS="${NEED_GPUS:-4}"
MEM_USED_THRESH_MIB="${MEM_USED_THRESH_MIB:-1000}"
POLL_SECS="${POLL_SECS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
STABLE_GAP_SECS="${STABLE_GAP_SECS:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/combined_qwen3_training.sh}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found" >&2
    exit 1
fi

ts() { date '+%F %T'; }

# Print indices of GPUs that have NO compute processes and used mem < threshold, sorted ascending.
free_gpus() {
    local busy_uuids
    busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)"
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/, */,/g' \
        | while IFS=',' read -r idx uuid memused; do
            [ -z "${idx:-}" ] && continue
            if ! grep -q "$uuid" <<<"$busy_uuids" && [ "${memused:-0}" -lt "$MEM_USED_THRESH_MIB" ]; then
                echo "$idx"
            fi
        done | sort -n
}

echo "[$(ts)] waiting for ${NEED_GPUS} free GPU(s) (used mem < ${MEM_USED_THRESH_MIB} MiB, no compute procs)..."
SELECTED=""
while true; do
    mapfile -t FREE < <(free_gpus)
    if [ "${#FREE[@]}" -ge "$NEED_GPUS" ]; then
        candidate="${FREE[*]:0:$NEED_GPUS}"
        # Stability re-checks: make sure the same GPUs stay free across a few polls so we don't
        # grab GPUs another job is in the middle of claiming.
        stable=1
        for ((c = 1; c < STABLE_CHECKS; c++)); do
            sleep "$STABLE_GAP_SECS"
            mapfile -t FREE2 < <(free_gpus)
            recheck="${FREE2[*]:0:$NEED_GPUS}"
            if [ "${#FREE2[@]}" -lt "$NEED_GPUS" ] || [ "$recheck" != "$candidate" ]; then
                stable=0
                break
            fi
        done
        if [ "$stable" -eq 1 ]; then
            SELECTED="$(echo "$candidate" | tr ' ' ',')"
            break
        fi
        echo "[$(ts)] free set changed during stability check; continuing to wait..."
    else
        echo "[$(ts)] ${#FREE[@]}/${NEED_GPUS} free (${FREE[*]:-none}); retry in ${POLL_SECS}s"
    fi
    sleep "$POLL_SECS"
done

export CUDA_VISIBLE_DEVICES="$SELECTED"
echo "[$(ts)] launching training on GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[$(ts)] script: ${TRAIN_SCRIPT}"
exec bash "$TRAIN_SCRIPT" "$@"
