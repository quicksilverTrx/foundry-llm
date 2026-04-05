# syntax=docker/dockerfile:1.7
#
# Base image: vastai/pytorch (mini-py311 variant — ~4.8 GB compressed, ~9 GB on disk)
# This replaces the runpod/pytorch:devel image (~9 GB compressed, ~17 GB on disk).
#
# On Vast.ai, using `vastai/pytorch:@vastai-automatic-tag` as the template image
# lets Vast.ai pre-cache the base and only the diff layers are stored in the overlay.
# When building locally for Docker Hub push, we pin to a specific tag.
#
# Tag convention: <torch>-<cu_toolkit>-<cuda_runtime>-mini-<python>-<date>
# Selected: 2.11.0-cu130-cuda-13.2-mini-py311  (PyTorch 2.11.0, Python 3.11, minimal CUDA)
# Fallback: if Vast.ai auto-resolves a different tag, SDPA will still work as long
#           as the GPU is CUDA 12.0+ (RTX 4090, A100, H100 all qualify).

FROM vastai/pytorch:2.11.0-cu130-cuda-13.2-mini-py311-2026-03-26

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.hf_cache \
    HF_DATASETS_CACHE=/workspace/.hf_cache/datasets \
    TMPDIR=/workspace/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/venv/main/bin:$PATH"

RUN mkdir -p /workspace/tmp /var/run/sshd /run/sshd && \
    chmod 755 /run/sshd /var/run/sshd && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      rsync \
      openssh-server && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/foundry-llm

# Install Python dependencies into the vastai venv (/venv/main — already has PyTorch 2.11+CUDA)
# PATH is set to /venv/main/bin first so pip/python resolve to the venv, not system Python 3.12
# Matches the RunPod thin-image spec (docker_runpod_pipeline_implementation_2026-03-22.md):
#   numpy==1.26.4      — shard memmap, pretokenization, array ops
#   sentencepiece      — SP16K tokenizer training + encoding
#   datasets           — FineWebEdu download and shard preparation
#   huggingface_hub    — HF auth + dataset streaming on a fresh pod
#   pytest             — in-image sanity checks
RUN pip install --no-cache-dir \
      "numpy==1.26.4" \
      "sentencepiece==0.2.0" \
      "datasets>=2.20.0" \
      "huggingface_hub" \
      pytest

COPY . /app/foundry-llm

RUN mkdir -p /var/run/sshd /workspace /root/.ssh && \
    chmod 700 /root/.ssh && \
    printf '\nPermitRootLogin yes\nPubkeyAuthentication yes\nPasswordAuthentication no\nUsePAM no\nStrictModes no\n' >> /etc/ssh/sshd_config && \
    pip install -e . --no-deps && \
    ln -sf /venv/main/bin/python3 /usr/local/bin/python 2>/dev/null || true

# Optional: bake in HF auth token (build secret — never stored in image layers)
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /root/.cache/huggingface && \
    if [ -f /run/secrets/hf_token ] && [ -s /run/secrets/hf_token ]; then \
      HF_TOKEN="$(tr -d '\r\n' < /run/secrets/hf_token)" && \
      export HF_TOKEN && \
      python3 -c "from huggingface_hub import login; import os; token = os.environ.get('HF_TOKEN', '').strip(); login(token=token, add_to_git_credential=False) if token else None; print('HF auth baked in.' if token else 'No HF token provided.')" ; \
    else \
      echo "No Hugging Face build secret provided; image will require runtime HF auth."; \
    fi

EXPOSE 22

ENTRYPOINT ["/app/foundry-llm/bin/runpod_container_entrypoint.sh"]
CMD ["bash", "-lc", "mkdir -p /workspace/foundry-llm-runtime/logs /workspace/tmp /workspace/.hf_cache/datasets && tail -f /dev/null"]
