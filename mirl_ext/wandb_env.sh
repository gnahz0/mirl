# Source after activating Python, from the repository root.
# Share explicit credentials and identity verification across training stages.

# Jobs submitted before this setting was added may have an older environment.
if [[ -z "${WANDB_EXPECTED_USERNAME:-}" && -r mirl.env ]]; then
    source mirl.env
fi
export WANDB_EXPECTED_USERNAME="${WANDB_EXPECTED_USERNAME:?set it in private mirl.env}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${MIRL_CLUSTER_ROOT:?}/.wandb_key}"
export WANDB_ENTITY="${MIRL_WANDB_ENTITY:?}" WANDB_MODE=online
if [[ ! -r "$WANDB_API_KEY_FILE" ]]; then
    echo "missing W&B key: $WANDB_API_KEY_FILE" >&2
    return 1
fi
export WANDB_API_KEY="$(<"$WANDB_API_KEY_FILE")"

# Never fall back to another user's netrc on a shared cluster account.
unset NETRC
if [[ -r "$MIRL_CLUSTER_ROOT/.netrc" ]]; then
    export NETRC="$MIRL_CLUSTER_ROOT/.netrc"
fi
"${PYTHON:-python}" - <<'PY'
from verl.utils.tracking import configure_wandb_auth

print(f"W&B credential verified: {configure_wandb_auth()}")
PY
