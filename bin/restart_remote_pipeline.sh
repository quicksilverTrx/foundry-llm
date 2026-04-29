#!/usr/bin/env bash
set -euo pipefail

# Stop the live remote pipeline/training processes and start a fresh command in
# the same pod. This keeps the prepared workspace intact while applying updated
# code already synced into the container.

usage() {
  cat <<'EOF'
Usage:
  REMOTE_HOST=<host> [MODE=auto] [EXTRA_PIPELINE_ARGS="..."] \
    [REMOTE_USER=root] [REMOTE_PORT=22] \
    [IDENTITY_FILE=~/.ssh/foundry_llm_remote] \
    [REMOTE_REPO_DIR=/app/foundry-llm] \
    [WORKSPACE_ROOT=/workspace/foundry-llm-runtime] \
    ./bin/restart_remote_pipeline.sh

Optional direct command override:
  REMOTE_HOST=<host> DIRECT_COMMAND="python scripts/pretrain_nanollama.py --config ..." \
    ./bin/restart_remote_pipeline.sh

Environment:
  REMOTE_HOST           Required. SSH host or gateway target.
  MODE                  Pipeline mode for runpod_pipeline.py. Default: auto
  EXTRA_PIPELINE_ARGS   Extra args appended after --workspace-root.
  DIRECT_COMMAND        If set, runs this command instead of the pipeline mode.
  REMOTE_USER           Remote SSH user. Default: root
  REMOTE_PORT           Remote SSH port. Default: 22
  IDENTITY_FILE         SSH private key path. Default: ~/.ssh/foundry_llm_remote
  REMOTE_REPO_DIR       Remote repo path. Default: /app/foundry-llm
  WORKSPACE_ROOT        Persistent runtime root. Default: /workspace/foundry-llm-runtime
  EXTRA_SSH_ARGS        Extra args appended to the ssh command.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"
IDENTITY_FILE="${IDENTITY_FILE:-${HOME}/.ssh/foundry_llm_remote}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/app/foundry-llm}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/foundry-llm-runtime}"
MODE="${MODE:-auto}"
EXTRA_PIPELINE_ARGS="${EXTRA_PIPELINE_ARGS:-}"
DIRECT_COMMAND="${DIRECT_COMMAND:-}"
EXTRA_SSH_ARGS="${EXTRA_SSH_ARGS:-}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "Set REMOTE_HOST to the remote pod hostname or SSH gateway target." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${IDENTITY_FILE}" ]]; then
  echo "SSH identity file not found: ${IDENTITY_FILE}" >&2
  exit 1
fi

ssh_cmd=(
  ssh
  -i "${IDENTITY_FILE}"
  -p "${REMOTE_PORT}"
  -o StrictHostKeyChecking=accept-new
)
if [[ -n "${EXTRA_SSH_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_ssh_parts=( ${EXTRA_SSH_ARGS} )
  ssh_cmd+=( "${extra_ssh_parts[@]}" )
fi

if [[ -n "${DIRECT_COMMAND}" ]]; then
  remote_command="ulimit -n 65535 || true; ${DIRECT_COMMAND}"
else
  remote_command="ulimit -n 65535 || true; python scripts/runpod_pipeline.py --mode ${MODE} --workspace-root ${WORKSPACE_ROOT}"
  if [[ -n "${EXTRA_PIPELINE_ARGS}" ]]; then
    remote_command="${remote_command} ${EXTRA_PIPELINE_ARGS}"
  fi
fi

"${ssh_cmd[@]}" "${REMOTE_USER}@${REMOTE_HOST}" /bin/bash <<EOF
set -euo pipefail
pkill -f "runpod_pipeline.py|pretrain_nanollama.py|prepare_dataset.py" || true
cd '${REMOTE_REPO_DIR}'
mkdir -p '${WORKSPACE_ROOT}/logs'
nohup bash -lc "$(printf "%q" "${remote_command}") >> '${WORKSPACE_ROOT}/logs/pipeline.log' 2>&1" >/dev/null 2>&1 &
printf 'Started remote command: %s\n' "${remote_command}"
EOF
