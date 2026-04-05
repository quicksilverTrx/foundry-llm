#!/usr/bin/env bash
set -euo pipefail

# Build the current image and exercise the remote-like startup path. This is a
# stronger gate than verify-local because it checks the real streaming auth
# branch used when the operator explicitly starts the full pipeline inside the
# container.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/bin/runpod_journal_lib.sh"

IMAGE_NAME="${IMAGE_NAME:-foundry-llm-runpod}"
IMAGE_TAG="${IMAGE_TAG:-remote-startup-verify}"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${REPO_ROOT}/HF_Token.txt}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/.docker-remote-startup}"

CURRENT_STAGE="remote startup verify"
CURRENT_COMMAND="bin/verify_remote_startup.sh"
ERROR_TRAP='status=$?; append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "failed" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, work_dir=${WORK_DIR}" "exit_status=${status}"; exit "${status}"'
trap "${ERROR_TRAP}" ERR

PLATFORM_ARGS=()
if [[ "$(uname -m)" == "arm64" ]]; then
  PLATFORM_ARGS=(--platform linux/amd64)
fi

mkdir -p "${WORK_DIR}"

BUILD_ARGS=("${PLATFORM_ARGS[@]}")
if [[ -f "${HF_TOKEN_FILE}" ]]; then
  BUILD_ARGS+=(--secret "id=hf_token,src=${HF_TOKEN_FILE}")
fi

DOCKER_BUILDKIT=1 docker build "${BUILD_ARGS[@]}" -t "${IMAGE_REF}" .

MISSING_AUTH_LOG="${WORK_DIR}/missing_auth.log"
trap - ERR
set +e
docker run --rm \
  "${PLATFORM_ARGS[@]}" \
  -e HF_HOME=/tmp/hf-empty \
  -e HF_TOKEN= \
  "${IMAGE_REF}" \
  bash -lc 'mkdir -p /tmp/hf-empty && cd /app/foundry-llm && python scripts/p15_runpod_pipeline.py --mode auto --workspace-root /workspace/foundry-llm-runtime' \
  > "${MISSING_AUTH_LOG}" 2>&1
MISSING_AUTH_STATUS=$?
set -e
trap "${ERROR_TRAP}" ERR

if [[ "${MISSING_AUTH_STATUS}" -eq 0 ]]; then
  echo "Expected remote startup auth check to fail without HF auth, but it succeeded." >&2
  exit 1
fi
grep -q "Hugging Face auth is required for streaming runs" "${MISSING_AUTH_LOG}"
if grep -q "huggingface_hub is required for runtime HF auth" "${MISSING_AUTH_LOG}"; then
  echo "Observed deprecated huggingface_hub startup failure again." >&2
  exit 1
fi

if [[ -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN_VALUE="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
  AUTH_OK_LOG="${WORK_DIR}/auth_ok.log"
  docker run --rm \
    "${PLATFORM_ARGS[@]}" \
    -e HF_TOKEN="${HF_TOKEN_VALUE}" \
    "${IMAGE_REF}" \
    bash -lc "cd /app/foundry-llm && python -c \"import sys; sys.path.insert(0, '/app/foundry-llm/scripts'); import p15_runpod_pipeline as pipeline; pipeline._initialize_hf_auth(require_token=True); print('HF_AUTH_OK')\"" \
    > "${AUTH_OK_LOG}" 2>&1
  grep -q "HF_AUTH_OK" "${AUTH_OK_LOG}"
fi

printf 'Remote startup verification passed for %s\n' "${IMAGE_REF}"
printf 'Missing-auth log: %s\n' "${MISSING_AUTH_LOG}"
if [[ -f "${WORK_DIR}/auth_ok.log" ]]; then
  printf 'Auth-ok log: %s\n' "${WORK_DIR}/auth_ok.log"
fi

append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "success" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, work_dir=${WORK_DIR}" "missing_auth_log=${MISSING_AUTH_LOG}"
trap - ERR
