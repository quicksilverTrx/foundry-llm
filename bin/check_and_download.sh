#!/usr/bin/env bash
set -euo pipefail

# Legacy RunPod-only helper: discover a pod via the RunPod API, tail its log
# over SSH when reachable, and download the newest bundle when it exists.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/bin/runpod_journal_lib.sh"

RUNPOD_API_KEY_FILE="${RUNPOD_API_KEY_FILE:-${REPO_ROOT}/runpod_api_key}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
if [[ -z "${RUNPOD_API_KEY}" ]] && [[ -f "${RUNPOD_API_KEY_FILE}" ]]; then
  RUNPOD_API_KEY="$(tr -d '\r\n' < "${RUNPOD_API_KEY_FILE}")"
fi
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY or RUNPOD_API_KEY_FILE}"
RUNPOD_API_URL="${RUNPOD_API_URL:-https://rest.runpod.io/v1}"
STATE_FILE="${STATE_FILE:-}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${REPO_ROOT}/downloads/runpod}"
REMOTE_BUNDLE_NAME="${REMOTE_BUNDLE_NAME:-}"
IDENTITY_FILE="${IDENTITY_FILE:-${HOME}/.ssh/github}"
EXTRACT_BUNDLE="${EXTRACT_BUNDLE:-1}"

CURRENT_STAGE="monitor and download"
CURRENT_COMMAND="bin/check_and_download.sh"
JOURNAL_RECORDED=0
trap 'status=$?; append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "failed" "${CURRENT_COMMAND}" "state_file=${STATE_FILE}, identity_file=${IDENTITY_FILE}" "exit_status=${status}"; exit "${status}"' ERR

if [[ -z "${STATE_FILE}" ]]; then
  # Reuse the first saved RunPod state file from a prior API-created pod.
  STATE_FILE="$(find .runpod -maxdepth 1 -name '*.state.json' | head -n 1 || true)"
fi
STATE_FILE="${STATE_FILE:?Set STATE_FILE or create a pod with bin/ship_and_train.sh}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

POD_ID="$(python3 - <<'PY' "${STATE_FILE}"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["pod_id"])
PY
)"

WORKSPACE_ROOT="$(python3 - <<'PY' "${STATE_FILE}"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload.get("workspace_root", "/workspace/foundry-llm-runtime"))
PY
)"

POD_JSON="$(curl --silent --show-error --fail \
  --request GET \
  --url "${RUNPOD_API_URL}/pods/${POD_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}")"

printf '%s\n' "${POD_JSON}" > ".runpod/${POD_ID}.latest.json"

python3 - <<'PY' "${POD_JSON}"
import json
import sys
payload = json.loads(sys.argv[1])
port_mappings = payload.get("portMappings") or {}
ssh_port = None
if isinstance(port_mappings, dict):
    ssh_port = port_mappings.get("22")
print(f"pod_id={payload.get('id')}")
print(f"desired_status={payload.get('desiredStatus')}")
print(f"runtime_status={payload.get('runtime', {}).get('uptimeInSeconds') if isinstance(payload.get('runtime'), dict) else None}")
print(f"public_ip={payload.get('publicIp')}")
print(f"ssh_port={ssh_port}")
PY

SSH_TARGET="$(python3 - <<'PY' "${POD_JSON}"
import json
import sys
payload = json.loads(sys.argv[1])
port_mappings = payload.get("portMappings") or {}
port = port_mappings.get("22") if isinstance(port_mappings, dict) else None
ip = payload.get("publicIp")
if ip and port:
    print(f"root@{ip}:{port}")
PY
)"

if [[ -n "${SSH_TARGET}" ]] && [[ -f "${IDENTITY_FILE}" ]]; then
  SSH_HOST="${SSH_TARGET%:*}"
  SSH_PORT="${SSH_TARGET##*:}"
  # Tail the current pipeline log first so you can see whether the remote run
  # is still progressing before attempting bundle download.
  ssh -o StrictHostKeyChecking=no -i "${IDENTITY_FILE}" -p "${SSH_PORT}" "${SSH_HOST}" \
    "tail -n 120 '${WORKSPACE_ROOT}/logs/pipeline.log' 2>/dev/null || true"

  mkdir -p "${DOWNLOAD_DIR}"
  if [[ -z "${REMOTE_BUNDLE_NAME}" ]]; then
    REMOTE_BUNDLE_NAME="$(ssh -o StrictHostKeyChecking=no -i "${IDENTITY_FILE}" -p "${SSH_PORT}" "${SSH_HOST}" \
      "ls -1t '${WORKSPACE_ROOT}/bundles'/*.tar.gz 2>/dev/null | head -n 1" || true)"
  else
    REMOTE_BUNDLE_NAME="${WORKSPACE_ROOT}/bundles/${REMOTE_BUNDLE_NAME}"
  fi

  if [[ -n "${REMOTE_BUNDLE_NAME}" ]]; then
    LOCAL_BUNDLE_PATH="${DOWNLOAD_DIR}/$(basename "${REMOTE_BUNDLE_NAME}")"
    # Pull the newest completed artifact bundle back to the local machine.
    scp -o StrictHostKeyChecking=no -i "${IDENTITY_FILE}" -P "${SSH_PORT}" \
      "${SSH_HOST}:${REMOTE_BUNDLE_NAME}" "${LOCAL_BUNDLE_PATH}"
    printf 'Downloaded bundle to %s\n' "${LOCAL_BUNDLE_PATH}"

    if [[ "${EXTRACT_BUNDLE}" == "1" ]]; then
      EXTRACT_DIR="${DOWNLOAD_DIR}/$(basename "${LOCAL_BUNDLE_PATH}" .tar.gz)"
      rm -rf "${EXTRACT_DIR}"
      mkdir -p "${EXTRACT_DIR}"
      tar -xzf "${LOCAL_BUNDLE_PATH}" -C "${EXTRACT_DIR}"
      printf 'Extracted bundle to %s\n' "${EXTRACT_DIR}"
      # Print the nominal best-checkpoint path to reduce manual digging.
      find "${EXTRACT_DIR}" -path '*/checkpoints/best_val.pt' -print 2>/dev/null || true
    fi

    append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "success" "${CURRENT_COMMAND}" "pod_id=${POD_ID}, identity_file=${IDENTITY_FILE}" "bundle=${LOCAL_BUNDLE_PATH}"
    JOURNAL_RECORDED=1
    trap - ERR
  fi
fi

if [[ "${JOURNAL_RECORDED}" == "0" ]]; then
  append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "in_progress" "${CURRENT_COMMAND}" "pod_id=${POD_ID}, identity_file=${IDENTITY_FILE}" "bundle_not_available_yet_or_ssh_unreachable"
fi
trap - ERR
