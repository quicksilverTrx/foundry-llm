#!/usr/bin/env bash
# setup_pod.sh — Bootstrap script for SwiftLlama-350M ablation run on H100 pod.
#
# Run this ONCE after the pod starts:
#   bash /app/foundry-llm/scripts/setup_pod.sh 2>&1 | tee /workspace/setup.log
#
# What it does:
#   1. Creates /workspace dirs
#   2. Pulls the correct git branch into /app/foundry-llm
#   3. Verifies GPU + code (pytest)
#   4. Downloads + tokenizes FineWebEDU 10BT in parallel
#   5. Writes a READY file when done
#
# Expected env vars (set by Vast.ai template or passed manually):
#   HF_TOKEN   — HuggingFace token for dataset download
#
# After this script succeeds, start the ablation probes with:
#   python3 /app/foundry-llm/scripts/run_ablation_probes.py \
#       --data-dir /workspace/edu_fineweb10B \
#       --runs-dir /workspace/runs/ablation \
#       --skip-existing
set -euo pipefail

REPO=/app/foundry-llm
BRANCH="${BRANCH:-main}"
REMOTE=https://github.com/quicksilverTrx/foundry-llm.git
DATA_DIR=/workspace/edu_fineweb10B
LOG=/workspace/setup.log
READY=/workspace/READY

mkdir -p /workspace/logs /workspace/runs/ablation /workspace/runs/production

echo "============================================================"
echo " SwiftLlama-350M Pod Setup"
echo " $(date -u)"
echo "============================================================"

# ── 1. GPU check ──────────────────────────────────────────────────────────────
echo "[1/5] GPU verification"
nvidia-smi | head -8
python3 -c "
import torch
d = torch.cuda.get_device_name(0)
v = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {d}  VRAM: {v:.1f} GB')
assert v > 70, f'Expected H100 80GB, got {v:.1f}GB'
print('GPU check: PASSED')
"

# ── 2. Pull latest code ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Pulling code from branch ${BRANCH}"
cd "${REPO}"
# In case the repo was cloned at image build time, fetch and switch branch.
git remote set-url origin "${REMOTE}" 2>/dev/null || true
git fetch origin "${BRANCH}" 2>&1 | tail -5
git checkout "${BRANCH}" 2>&1 || git checkout -b "${BRANCH}" "origin/${BRANCH}"
git pull origin "${BRANCH}" 2>&1 | tail -5
echo "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format='%s')"

# ── 3. Install deps ────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing Python package in editable mode"
pip install -e . --quiet
# Make sure datasets and tiktoken are available (already in image, but belt-and-suspenders)
pip install datasets tiktoken huggingface_hub --quiet

# ── 4. Run tests ───────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Running test suite"
cd "${REPO}"
python3 -m pytest tests/ -x -q 2>&1 | tail -5
echo "Tests: PASSED"

# ── 5. Download + tokenize FineWebEDU 10BT ────────────────────────────────────
echo ""
echo "[5/5] Downloading + tokenizing FineWebEDU-10BT"
echo "      Output dir: ${DATA_DIR}"
echo "      This takes ~20-40 minutes with 8 parallel tokenizers."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Download may fail for gated datasets."
fi

DOWNLOAD_WORKERS=8
TOKENIZE_WORKERS=8

python3 "${REPO}/scripts/download_fineweb_parallel.py" \
    --out-dir        "${DATA_DIR}" \
    --tokenize-workers "${TOKENIZE_WORKERS}" \
    --hf-token       "${HF_TOKEN:-}" \
    2>&1 | tee /workspace/logs/fineweb_download.log

# Verify shards
TRAIN_SHARDS=$(ls "${DATA_DIR}"/edufineweb_train_*.npy 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "${DATA_DIR}"/edufineweb_val_*.npy 2>/dev/null | wc -l)
echo "Train shards: ${TRAIN_SHARDS}"
echo "Val   shards: ${VAL_SHARDS}"

if [[ "${TRAIN_SHARDS}" -lt 50 ]]; then
    echo "ERROR: Expected ~99 train shards, got ${TRAIN_SHARDS}. Check download log."
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " SETUP COMPLETE  $(date -u)"
echo " Train shards: ${TRAIN_SHARDS}   Val shards: ${VAL_SHARDS}"
echo "============================================================"
echo ""
echo "Next step — start ablation probes (run in tmux):"
echo ""
echo "  tmux new-session -d -s ablation"
echo "  tmux send-keys -t ablation \\"
echo "    'python3 ${REPO}/scripts/run_ablation_probes.py \\"
echo "        --data-dir ${DATA_DIR} \\"
echo "        --runs-dir /workspace/runs/ablation' Enter"
echo ""

date -u > "${READY}"
echo "READY file written to ${READY}"
