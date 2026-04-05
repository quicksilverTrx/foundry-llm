#!/usr/bin/env bash
set -euo pipefail

# Sync the local repo tree into a running remote pod over plain SSH by streaming
# a compressed tarball over stdin. This avoids pod recreation and does not rely
# on scp/rsync availability.

usage() {
  cat <<'EOF'
Usage:
  REMOTE_HOST=<host> [REMOTE_USER=root] [REMOTE_PORT=22] \
    [IDENTITY_FILE=~/.ssh/foundry_llm_remote] \
    [REMOTE_REPO_DIR=/app/foundry-llm] \
    [REINSTALL_EDITABLE=0] \
    ./bin/sync_remote_code.sh

Environment:
  REMOTE_HOST          Required. SSH host or gateway target.
  REMOTE_USER          Remote SSH user. Default: root
  REMOTE_PORT          Remote SSH port. Default: 22
  IDENTITY_FILE        SSH private key path. Default: ~/.ssh/foundry_llm_remote
  REMOTE_REPO_DIR      Remote repo path to overwrite. Default: /app/foundry-llm
  REINSTALL_EDITABLE   Set to 1 to run 'python -m pip install -e . --no-deps'
                       after syncing. Default: 0
  EXTRA_SSH_ARGS       Extra args appended to the ssh command.

This script excludes large runtime/output directories so it only syncs source,
configs, scripts, and lightweight repo metadata.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"
IDENTITY_FILE="${IDENTITY_FILE:-${HOME}/.ssh/foundry_llm_remote}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/app/foundry-llm}"
REINSTALL_EDITABLE="${REINSTALL_EDITABLE:-0}"
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
  # Allow callers to pass one or more additional ssh flags via a single env var.
  # shellcheck disable=SC2206
  extra_ssh_parts=( ${EXTRA_SSH_ARGS} )
  ssh_cmd+=( "${extra_ssh_parts[@]}" )
fi

cd "${REPO_ROOT}"

# Avoid emitting macOS xattr/provenance metadata that GNU tar on the remote
# side reports as unknown extended header keywords.
COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.venv' \
  --exclude='.docker-local' \
  --exclude='.docker-remote-startup' \
  --exclude='.runpod' \
  --exclude='downloads' \
  --exclude='artifacts' \
  --exclude='data' \
  --exclude='logs' \
  -czf - . \
| "${ssh_cmd[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "mkdir -p '${REMOTE_REPO_DIR}' && tar -xzf - -C '${REMOTE_REPO_DIR}'"

if [[ "${REINSTALL_EDITABLE}" == "1" ]]; then
  "${ssh_cmd[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd '${REMOTE_REPO_DIR}' && python -m pip install -e . --no-deps"
fi

printf 'Synced repo to %s@%s:%s\n' "${REMOTE_USER}" "${REMOTE_HOST}" "${REMOTE_REPO_DIR}"
if [[ "${REINSTALL_EDITABLE}" == "1" ]]; then
  printf 'Reinstalled editable package on remote host.\n'
fi
