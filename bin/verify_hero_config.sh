#!/usr/bin/env bash
set -euo pipefail

# Build the current image locally and run the tiny verify-local pipeline inside
# it. This is the fastest end-to-end container smoke check before a real remote
# run.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/bin/runpod_journal_lib.sh"

IMAGE_NAME="${IMAGE_NAME:-foundry-llm-runpod}"
IMAGE_TAG="${IMAGE_TAG:-local-verify}"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/.docker-local}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${REPO_ROOT}/HF_Token.txt}"

CURRENT_STAGE="local docker verify"
CURRENT_COMMAND="bin/verify_hero_config.sh"
trap 'status=$?; append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "failed" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, workspace_dir=${WORKSPACE_DIR}" "exit_status=${status}"; exit "${status}"' ERR

PLATFORM_ARGS=()
if [[ "$(uname -m)" == "arm64" ]]; then
  # Build x86_64 images from Apple Silicon so the result matches the remote GPU
  # providers that expect linux/amd64 containers.
  PLATFORM_ARGS=(--platform linux/amd64)
fi

mkdir -p "${WORKSPACE_DIR}"

BUILD_ARGS=("${PLATFORM_ARGS[@]}")
if [[ -f "${HF_TOKEN_FILE}" ]]; then
  # Include HF auth at build time when available so the image matches the
  # remote streaming path, even though verify-local itself does not need it.
  BUILD_ARGS+=(--secret "id=hf_token,src=${HF_TOKEN_FILE}")
fi

DOCKER_BUILDKIT=1 docker build "${BUILD_ARGS[@]}" -t "${IMAGE_REF}" .

# verify-local keeps the run tiny and CPU-safe while exercising prepare,
# preflight, smoke, and bundling.
docker run --rm \
  "${PLATFORM_ARGS[@]}" \
  -v "${WORKSPACE_DIR}:/workspace" \
  "${IMAGE_REF}" \
  python scripts/p15_runpod_pipeline.py --mode verify-local --workspace-root /workspace/foundry-llm-runtime

append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "success" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, workspace_dir=${WORKSPACE_DIR}" "verify_workspace=/workspace/foundry-llm-runtime"
trap - ERR
