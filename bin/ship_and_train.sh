#!/usr/bin/env bash
set -euo pipefail

# Legacy RunPod-only helper: create a pod through the RunPod API and point it
# at the already-built image. Manual provider-UI instance creation is the
# recommended path; keep this script only for explicit API-driven launches.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/bin/runpod_journal_lib.sh"

RUNPOD_API_KEY_FILE="${RUNPOD_API_KEY_FILE:-${REPO_ROOT}/runpod_api_key}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
if [[ -z "${RUNPOD_API_KEY}" ]] && [[ -f "${RUNPOD_API_KEY_FILE}" ]]; then
  RUNPOD_API_KEY="$(tr -d '\r\n' < "${RUNPOD_API_KEY_FILE}")"
fi
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY or RUNPOD_API_KEY_FILE}"
IMAGE_NAME="${IMAGE_NAME:-$(cat .runpod/latest-image.txt 2>/dev/null || true)}"
IMAGE_NAME="${IMAGE_NAME:?Set IMAGE_NAME or run bin/push_to_registry.sh first}"

RUNPOD_API_URL="${RUNPOD_API_URL:-https://rest.runpod.io/v1}"
RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-foundry-llm-hero}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
RUNPOD_GPU_TYPES="${RUNPOD_GPU_TYPES:-}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-100}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-20}"
RUNPOD_MIN_VCPU_PER_GPU="${RUNPOD_MIN_VCPU_PER_GPU:-9}"
RUNPOD_MIN_RAM_PER_GPU="${RUNPOD_MIN_RAM_PER_GPU:-50}"
RUNPOD_PUBLIC_KEY="${RUNPOD_PUBLIC_KEY:-}"
RUNPOD_PUBLIC_KEY_FILE="${RUNPOD_PUBLIC_KEY_FILE:-${HOME}/.ssh/github.pub}"
RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/foundry-llm-runtime}"

CURRENT_STAGE="pod create"
CURRENT_COMMAND="bin/ship_and_train.sh"
trap 'status=$?; append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "failed" "${CURRENT_COMMAND}" "image_name=${IMAGE_NAME}, gpu_types=${RUNPOD_GPU_TYPES}, workspace_root=${WORKSPACE_ROOT}" "exit_status=${status}"; exit "${status}"' ERR

RUNPOD_GPU_TYPES="${RUNPOD_GPU_TYPES:?Set RUNPOD_GPU_TYPES explicitly. This legacy RunPod launcher no longer hardcodes a GPU type.}"

if [[ -z "${RUNPOD_PUBLIC_KEY}" ]]; then
  # Prefer an explicit provider key file, then fall back to common local SSH
  # keys for convenience in legacy API-based launches.
  if [[ -f "${RUNPOD_PUBLIC_KEY_FILE}" ]]; then
    RUNPOD_PUBLIC_KEY="$(cat "${RUNPOD_PUBLIC_KEY_FILE}")"
  elif [[ -f "${HOME}/.ssh/id_ed25519.pub" ]]; then
    RUNPOD_PUBLIC_KEY="$(cat "${HOME}/.ssh/id_ed25519.pub")"
  elif [[ -f "${HOME}/.ssh/id_rsa.pub" ]]; then
    RUNPOD_PUBLIC_KEY="$(cat "${HOME}/.ssh/id_rsa.pub")"
  else
    echo "Set RUNPOD_PUBLIC_KEY or RUNPOD_PUBLIC_KEY_FILE" >&2
    exit 1
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

mkdir -p .runpod

REQUEST_JSON="$(python3 - <<'PY'
import json
import os

gpu_types = [x.strip() for x in os.environ["RUNPOD_GPU_TYPES"].split(",") if x.strip()]
data_centers = [x.strip() for x in os.environ.get("RUNPOD_DATA_CENTER_IDS", "").split(",") if x.strip()]

payload = {
    "cloudType": os.environ["RUNPOD_CLOUD_TYPE"],
    "computeType": "GPU",
    "containerDiskInGb": int(os.environ["RUNPOD_CONTAINER_DISK_GB"]),
    "dockerStartCmd": [
        "bash",
        "-lc",
        # Keep the remote start command aligned with the image's normal auto
        # mode so legacy API launches behave like manual provider launches.
        "mkdir -p {root}/logs && python scripts/p15_runpod_pipeline.py --mode auto --workspace-root {root} 2>&1 | tee -a {root}/logs/pipeline.log".format(
            root=os.environ["WORKSPACE_ROOT"]
        ),
    ],
    "env": {
        "PUBLIC_KEY": os.environ["RUNPOD_PUBLIC_KEY"],
    },
    "gpuCount": 1,
    "gpuTypeIds": gpu_types,
    "gpuTypePriority": "availability",
    "imageName": os.environ["IMAGE_NAME"],
    "interruptible": False,
    "minRAMPerGPU": int(os.environ["RUNPOD_MIN_RAM_PER_GPU"]),
    "minVCPUPerGPU": int(os.environ["RUNPOD_MIN_VCPU_PER_GPU"]),
    "name": os.environ["RUNPOD_POD_NAME"],
    "ports": ["22/tcp"],
    "supportPublicIp": True,
    "volumeInGb": int(os.environ["RUNPOD_VOLUME_GB"]),
    "volumeMountPath": "/workspace",
}

network_volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
if network_volume_id:
    payload["networkVolumeId"] = network_volume_id

if data_centers:
    payload["dataCenterIds"] = data_centers
    payload["dataCenterPriority"] = "custom"

print(json.dumps(payload))
PY
)"

RESPONSE_PATH=".runpod/${RUNPOD_POD_NAME}.create.json"

curl --silent --show-error --fail \
  --request POST \
  --url "${RUNPOD_API_URL}/pods" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data "${REQUEST_JSON}" \
  > "${RESPONSE_PATH}"

POD_ID="$(python3 - <<'PY' "${RESPONSE_PATH}"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["id"])
PY
)"

STATE_PATH=".runpod/${RUNPOD_POD_NAME}.state.json"
python3 - <<'PY' "${STATE_PATH}" "${RESPONSE_PATH}" "${RUNPOD_API_URL}" "${WORKSPACE_ROOT}"
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
create_path = Path(sys.argv[2])
api_url = sys.argv[3]
workspace_root = sys.argv[4]
payload = json.loads(create_path.read_text(encoding="utf-8"))
state = {
    "pod_id": payload["id"],
    "pod_name": payload.get("name"),
    "workspace_root": workspace_root,
    "api_url": api_url,
    "image": payload.get("image"),
    "created_response_path": str(create_path),
}
state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
print(state_path)
PY

printf 'Created pod %s\n' "${POD_ID}"
printf 'State written to %s\n' "${STATE_PATH}"
printf 'Use bin/check_and_download.sh to monitor and fetch artifacts.\n'
append_runpod_journal_entry "${REPO_ROOT}" "${CURRENT_STAGE}" "success" "${CURRENT_COMMAND}" "pod_id=${POD_ID}, image_name=${IMAGE_NAME}, gpu_types=${RUNPOD_GPU_TYPES}" "state_file=${STATE_PATH}"
trap - ERR
