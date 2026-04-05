#!/usr/bin/env bash

runpod_journal_path() {
  local repo_root="${1:?repo_root is required}"
  printf '%s\n' "${JOURNAL_PATH:-${repo_root}/docs/runpod_deployment_journal.md}"
}

append_runpod_journal_entry() {
  local repo_root="${1:?repo_root is required}"
  local stage="${2:?stage is required}"
  local outcome="${3:?outcome is required}"
  local command_text="${4:-}"
  local key_inputs="${5:-}"
  local details="${6:-}"

  local journal_path
  journal_path="$(runpod_journal_path "${repo_root}")"
  mkdir -p "$(dirname "${journal_path}")"

  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  {
    printf '\n## %s — %s\n\n' "${timestamp}" "${stage}"
    printf -- '- outcome: `%s`\n' "${outcome}"
    if [[ -n "${command_text}" ]]; then
      printf -- '- command: `%s`\n' "${command_text}"
    fi
    if [[ -n "${key_inputs}" ]]; then
      printf -- '- key inputs: %s\n' "${key_inputs}"
    fi
    if [[ -n "${details}" ]]; then
      printf -- '- details: %s\n' "${details}"
    fi
  } >> "${journal_path}"
}
