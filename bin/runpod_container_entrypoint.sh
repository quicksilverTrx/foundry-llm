#!/usr/bin/env bash
set -euo pipefail

# Start an SSH daemon for optional remote inspection and retrieval, then hand
# off to the image's default command. The training process itself is started by
# CMD, not by this entrypoint script.
mkdir -p /var/run/sshd /run/sshd /root/.ssh
chmod 755 /var/run/sshd /run/sshd
chmod 700 /root/.ssh

# If the provider injected a public key, allow root SSH for debugging and
# manual artifact transfer. Prefer SSH_PUBLIC_KEY but keep PUBLIC_KEY for
# backward compatibility with older launch scripts.
ssh_public_key="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [[ -n "${ssh_public_key}" ]]; then
  printf '%s\n' "${ssh_public_key}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

# Prefer the repo Conda env when it exists so runtime commands use the same
# interpreter that local validation uses.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate foundry-llm
fi

exec "$@"
