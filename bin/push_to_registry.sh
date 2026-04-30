#!/usr/bin/env bash
set -euo pipefail

# Build the production image, optionally package HF auth via BuildKit secret,
# then push the resulting tag and record it for later manual deployment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/bin/runpod_journal_lib.sh"

DEFAULT_HF_TOKEN_FILE="${REPO_ROOT}/HF_Token.txt"
if [[ ! -f "${DEFAULT_HF_TOKEN_FILE}" ]]; then
  DEFAULT_HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
fi

IMAGE_REPO="${IMAGE_REPO:-docker.io/your-dockerhub-username/foundry-llm-runpod}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH_LATEST="${PUSH_LATEST:-0}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${DEFAULT_HF_TOKEN_FILE}}"

CURRENT_STAGE="image build and push"
CURRENT_COMMAND="bin/push_to_registry.sh"
trap 'status=$?; append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "failed" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, hf_token_file=${HF_TOKEN_FILE}" "exit_status=${status}"; exit "${status}"' ERR

PLATFORM_ARGS=()
if [[ "$(uname -m)" == "arm64" ]]; then
  # Publish the same linux/amd64 image that remote GPU providers will run.
  PLATFORM_ARGS=(--platform linux/amd64)
fi

if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
  echo "HF token file not found at ${HF_TOKEN_FILE}" >&2
  echo "Set HF_TOKEN_FILE to a local Hugging Face token file before building the RunPod image." >&2
  exit 1
fi

DOCKER_BUILDKIT=1 docker build \
  "${PLATFORM_ARGS[@]}" \
  --secret "id=hf_token,src=${HF_TOKEN_FILE}" \
  -t "${IMAGE_REF}" .
docker push "${IMAGE_REF}"

mkdir -p .runpod
# Preserve the last successful image ref so legacy helpers and docs can refer
# to one canonical pushed tag.
printf '%s\n' "${IMAGE_REF}" > .runpod/latest-image.txt

if [[ "${PUSH_LATEST}" == "1" ]]; then
  docker tag "${IMAGE_REF}" "${IMAGE_REPO}:latest"
  docker push "${IMAGE_REPO}:latest"
fi

printf 'Pushed image: %s\n' "${IMAGE_REF}"
append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "success" "${CURRENT_COMMAND}" "image_ref=${IMAGE_REF}, hf_token_file=${HF_TOKEN_FILE}" "latest_image_file=.runpod/latest-image.txt"
trap - ERR
