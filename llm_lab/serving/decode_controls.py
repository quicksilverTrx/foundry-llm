# llm_lab/serving/decode_controls.py
from __future__ import annotations

from typing import Any, Literal

from llm_lab.serving.kv_cache import apply_sliding_window

StopReason = Literal["eos", "stop_token", "max_new_tokens", "stop_string"]


def should_stop_tokens(
    generated_token_ids: list[int],
    *,
    eos_token_id: int | None,
    stop_token_ids: set[int] | None,
    max_new_tokens: int,
) -> tuple[bool, StopReason | None]:
    """Return (stop, reason) for token-level stop checks."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if len(generated_token_ids) >= max_new_tokens:
        return True, "max_new_tokens"
    if not generated_token_ids:
        return False, None
    last = generated_token_ids[-1]
    if eos_token_id is not None and last == eos_token_id:
        return True, "eos"
    if stop_token_ids is not None and last in stop_token_ids:
        return True, "stop_token"
    return False, None


def should_stop_text(
    decoded_text_so_far: str,
    *,
    stop_strings: list[str] | None,
) -> tuple[bool, StopReason | None]:
    """Return (stop, reason) for completion text stop checks."""
    if not stop_strings:
        return False, None
    for s in stop_strings:
        if not s:
            continue
        if s in decoded_text_so_far:
            return True, "stop_string"
    return False, None


def apply_context_truncation(
    prompt_ids: list[int],
    generated_ids: list[int],
    *,
    block_size: int,
) -> tuple[list[int], list[int]]:
    """Keep only the last block_size tokens across prompt+generated."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    prompt = list(prompt_ids)
    generated = list(generated_ids)
    full = prompt + generated
    if len(full) <= block_size:
        return prompt, generated

    kept = full[-block_size:]
    generated_len = len(generated)
    kept_len = len(kept)
    gen_keep = min(generated_len, kept_len)
    if gen_keep <= 0:
        return kept, []
    prompt_part = kept[:-gen_keep]
    generated_part = kept[-gen_keep:]
    return prompt_part, generated_part


def truncate_kv_cache_to_block_size(cache_state: Any, *, keep_last_n: int):
    """Truncate cache in-place-style and return updated state."""
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be >= 0")
    if cache_state.past_key_values is None:
        return cache_state

    truncated = apply_sliding_window(cache_state.past_key_values, max_len=keep_last_n)
    cache_state.past_key_values = truncated
    cache_state.past_len = int(truncated[0][0].shape[2]) if truncated else 0
    return cache_state
