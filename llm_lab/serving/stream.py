# llm_lab/serving/stream.py
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Iterator

import torch

from llm_lab.serving._shared import build_step_distribution, resolve_eos_token_id, token_logprobability
from llm_lab.serving.decode_controls import (
    apply_context_truncation,
    should_stop_text,
    should_stop_tokens,
    truncate_kv_cache_to_block_size,
)
from llm_lab.serving.sampling import select_next_token_id
from llm_lab.serving.schemas import GenerateRequest

if TYPE_CHECKING:
    from llm_lab.serving.engine import Engine

# Re-export for backward compat
StreamRequest = GenerateRequest


def sse_encode(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def iter_token_sse(*, engine: "Engine", req: StreamRequest) -> Iterator[str]:
    # Stream data contract:
    # - emit incremental `token` events
    # - emit exactly one terminal `final` event
    # - final event is authoritative for stop_reason + metrics
    if engine.block_size <= 0:
        raise ValueError("block_size must be > 0")

    # Stage 1: request normalization and prompt encoding for streaming path.
    t_request_start = time.perf_counter()
    prompt_ids_list = engine._ensure_prompt_ids(engine.tokenizer.encode(req.prompt))
    device = next(engine.model.parameters()).device

    # Stage 2: shape prompt for prefill ([1, T]) and initialize bookkeeping counters.
    prompt_tensor = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
    mask_tensor = torch.ones_like(prompt_tensor)

    start_prefill_calls = engine.prefill_calls
    start_decode_calls = engine.decode_calls

    # Stage 3: prefill once to seed decode loop and capture prompt latency.
    t_prefill_start = time.perf_counter()
    prefill_logits, cache, _ = engine.prefill(prompt_tensor, mask_tensor)
    prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0
    step_logits = prefill_logits[0]
    decode_ms_total = 0.0

    if hasattr(engine.model, "schedule"):
        bootstrap_id = torch.tensor([[prompt_ids_list[-1]]], dtype=torch.long, device=device)
        t_decode_start = time.perf_counter()
        step_logits, cache = engine.decode_step(bootstrap_id, cache)
        decode_ms_total += (time.perf_counter() - t_decode_start) * 1000.0
        step_logits = step_logits[0]

    generated_ids: list[int] = []
    generated_window: list[int] = []
    token_logprobs: list[float] = []
    token_counts: dict[int, int] = {}
    stop_reason: str | None = None

    # emitted_count tracks how many tokens are already sent over SSE.
    emitted_count = 0
    ttft_ms: float | None = None

    def _emit_new_tokens(prefix_ids: list[int]) -> Iterator[str]:
        nonlocal emitted_count, ttft_ms
        # Emits only the unseen suffix [emitted_count : len(prefix_ids)].
        for idx in range(emitted_count, len(prefix_ids)):
            tid = int(prefix_ids[idx])
            payload = {
                "index": idx,
                "token_id": tid,
                "token_text": engine.tokenizer.decode([tid]),
            }
            if req.return_logprobs and idx < len(token_logprobs):
                payload["token_logprob"] = float(token_logprobs[idx])
            if ttft_ms is None:
                # TTFT is stamped exactly when first token event is emitted.
                ttft_ms = (time.perf_counter() - t_request_start) * 1000.0
            yield sse_encode("token", payload)
        # Invariant: emitted_count only increases and tracks already-streamed token count.
        emitted_count = len(prefix_ids)

    # Streaming stop-string safety: keep a text holdback to avoid leaking partial marker tails.
    if req.stop_strings:
        holdback_chars = max((len(s) for s in req.stop_strings if s), default=0) - 1
        holdback_chars = max(holdback_chars, 0)
    else:
        holdback_chars = 0

    # Holdback protects against leaking partial stop-marker tails during streaming.

    eos_token_id = resolve_eos_token_id(engine.tokenizer)
    stop_token_ids = set(req.stop_token_ids) if req.stop_token_ids else None

    for step_idx in range(req.max_new_tokens):
        # Stage 4: sample/select candidate token for this decode step.
        if req.return_logprobs:
            step_distribution = build_step_distribution(
                    step_logits,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=getattr(req, "repetition_penalty", None),
                    frequency_penalty=getattr(req, "frequency_penalty", None),
                    generated_token_ids=generated_ids,
                    token_counts=token_counts,
                )
        cand_id = select_next_token_id(
            step_logits,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=None,
            frequency_penalty=None,
            generated_token_ids=generated_ids,
            token_counts=token_counts,
            seed=engine._derive_step_seed(req.seed, step_idx),
            greedy=req.temperature == 0.0,
        )

        trial = generated_ids + [cand_id]
        # Stop precedence: token-level stops are checked before accepting candidate token.
        stop_trial, reason_trial = should_stop_tokens(
            trial,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            max_new_tokens=req.max_new_tokens,
        )
        if stop_trial and reason_trial in {"eos", "stop_token"}:
            # Token-level stop exits before appending marker token.
            stop_reason = reason_trial
            break

        # Stage 5: append accepted token to generated state.
        generated_ids.append(cand_id)
        generated_window.append(cand_id)
        token_counts[cand_id] = token_counts.get(cand_id, 0) + 1
        if req.return_logprobs:
            token_logprobs.append(token_logprobability(step_distribution, cand_id))

        # Stage 6: text-level stop handling and optional trimming at earliest marker boundary.
        completion_text = engine.tokenizer.decode(generated_ids)
        stop_text, _ = should_stop_text(completion_text, stop_strings=req.stop_strings)
        if stop_text:
            # Text-level stop trims to earliest marker boundary, then stream emits truncated prefix.
            earliest = engine._find_earliest_stop_index(completion_text, req.stop_strings)
            completion_text = completion_text[:earliest]
            generated_ids = engine._truncate_to_text_prefix(generated_ids, completion_text)
            if req.return_logprobs:
                token_logprobs = token_logprobs[:len(generated_ids)]

            keep = min(len(generated_ids), engine.block_size)
            generated_window = generated_ids[-keep:]
            yield from _emit_new_tokens(generated_ids)
            stop_reason = "stop_string"
            break

        # Stage 7: emit only safe prefix (withhold trailing chars that might complete stop string).
        safe_until = len(completion_text) - holdback_chars
        if safe_until < 0:
            safe_until = 0
        safe_text = completion_text[:safe_until]
        safe_ids = engine._truncate_to_text_prefix(generated_ids, safe_text)
        yield from _emit_new_tokens(safe_ids)

        # Stage 8: one decode step forward to get logits for next token selection.
        next_input = torch.tensor([[cand_id]], dtype=torch.long, device=device)
        t_decode_start = time.perf_counter()
        step_logits, cache = engine.decode_step(next_input, cache)
        decode_ms_total += (time.perf_counter() - t_decode_start) * 1000.0
        step_logits = step_logits[0]

        # Stage 9: enforce context+cache truncation parity with non-stream generate path.
        prompt_ids_list, generated_window = apply_context_truncation(
            prompt_ids_list,
            generated_window,
            block_size=engine.block_size,
        )
        cache = truncate_kv_cache_to_block_size(cache, keep_last_n=engine.block_size)

    # Stage 10: finalization, flush buffered tokens, and emit one final summary event.
    if stop_reason is None:
        stop_reason = "max_new_tokens"

    # Flush any buffered tokens now that generation has ended.
    yield from _emit_new_tokens(generated_ids)

    if ttft_ms is None:
        ttft_ms = prefill_ms

    completion_text = engine.tokenizer.decode(generated_ids) if generated_ids else ""
    decode_ms_per_token = decode_ms_total / float(max(len(generated_ids) - 1, 1)) if generated_ids else 0.0
    tokens_per_sec = (1000.0 / decode_ms_per_token) if decode_ms_per_token > 0 else 0.0
    metrics = {
        "generated_tokens": len(generated_ids),
        "decode_steps": engine.decode_calls - start_decode_calls,
        "prefill_calls": engine.prefill_calls - start_prefill_calls,
        "decode_calls": engine.decode_calls - start_decode_calls,
        "cache_len": cache.past_len,
        "ttft_ms": float(ttft_ms),
        "prefill_ms": float(prefill_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_ms_per_token": float(decode_ms_per_token),
        "tokens_per_sec": float(tokens_per_sec),
        "requested_dtype": getattr(engine, "requested_dtype", "fp32"),
        "runtime_dtype": getattr(engine, "runtime_dtype", "fp32"),
        "runtime_quant_mode": getattr(engine, "quant_mode", None),
        "runtime_fallback_reason": getattr(engine, "runtime_fallback_reason", None),
    }

    final_payload = {
        "stop_reason": stop_reason,
        "completion_text": completion_text,
        "completion_token_ids": generated_ids,
        "metrics": metrics,
    }
    # Terminal ownership: API-layer observability parses this same `final` payload.
    if req.return_logprobs and token_logprobs:
        final_payload["token_logprobs"] = token_logprobs
    yield sse_encode("final", final_payload)
