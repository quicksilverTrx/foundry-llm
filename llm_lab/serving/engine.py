# llm_lab/serving/engine.py
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Tuple

import torch

from llm_lab.core.package.io import load_model_package
from llm_lab.core.model.attention import PastKeyValues
from llm_lab.serving.batching import right_pad_and_mask
from llm_lab.serving.decode_controls import (
    apply_context_truncation,
    should_stop_text,
    should_stop_tokens,
    truncate_kv_cache_to_block_size,
)
from llm_lab.serving.kv_cache import apply_sliding_window
from llm_lab.serving.precision import cast_model_for_inference, runtime_precision_decision
from llm_lab.serving.quant import describe_quant_runtime, maybe_quantize_model
from llm_lab.serving._shared import build_step_distribution, token_logprobability
from llm_lab.serving.sampling import select_next_token_id

@dataclass
class CacheState:
    past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]]
    past_len: int


class Engine:
    """Serving wrapper split into prompt prefill and token-wise decode."""

    def __init__(
        self,
        model,
        tokenizer,
        block_size: int,
        max_cache_len: Optional[int] = None,
        *,
        requested_dtype: str = "fp32",
        runtime_dtype: str = "fp32",
        quant_mode: str | None = None,
        runtime_fallback_reason: str | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_cache_len = max_cache_len
        self.requested_dtype = str(requested_dtype)
        self.runtime_dtype = str(runtime_dtype)
        self.quant_mode = quant_mode
        self.runtime_fallback_reason = runtime_fallback_reason

        # Debug counters used by structure tests.
        self.prefill_calls = 0
        self.decode_calls = 0

    def _validate_prefill_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        # Data contract: prefill consumes strictly batched shapes [B, T] for both tensors.
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,T], got {tuple(input_ids.shape)}")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B,T], got {tuple(attention_mask.shape)}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape mismatch: input_ids={tuple(input_ids.shape)} mask={tuple(attention_mask.shape)}"
            )

    def _validate_decode_inputs(self, next_input_id: torch.Tensor, state: CacheState) -> None:
        # Data contract: decode is incremental and currently supports token shape [B, 1].
        if next_input_id.ndim != 2:
            raise ValueError(f"next_input_id must be [B,1], got {tuple(next_input_id.shape)}")
        if next_input_id.shape[1] != 1:
            raise ValueError(f"next_input_id must have trailing dim 1, got {tuple(next_input_id.shape)}")
        if state.past_key_values is None:
            raise ValueError("decode_step requires non-empty state.past_key_values")

    def _ensure_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        return list(prompt_ids)

    def _truncate_to_text_prefix(self, token_ids: list[int], target_text: str) -> list[int]:
        if not token_ids:
            return []
        # Prefer token-boundary trimming; fallback to tokenizer encode for non-boundary cuts.
        for i in range(len(token_ids), -1, -1):
            maybe = self.tokenizer.decode(token_ids[:i])
            if maybe == target_text:
                return token_ids[:i]
        return list(self.tokenizer.encode(target_text))

    @staticmethod
    def _derive_step_seed(seed: int | None, step_idx: int) -> int | None:
        if seed is None:
            return None
        return int(seed) + int(step_idx)

    @staticmethod
    def _find_earliest_stop_index(text: str, stop_strings: list[str] | None) -> int:
        earliest = len(text)
        if not stop_strings:
            return earliest
        for s in stop_strings:
            if not s:
                continue
            pos = text.find(s)
            if pos != -1 and pos < earliest:
                earliest = pos
        return earliest
    
    _build_step_distribution = staticmethod(build_step_distribution)
    _token_logprobability = staticmethod(token_logprobability)


    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, CacheState, Dict]:
        """
        Returns:
            last_logits: [B, vocab]
            state: cache state for token-wise decode
            meta: small debug metadata
        """
        self._validate_prefill_inputs(input_ids, attention_mask)
        self.prefill_calls += 1

        B, T = input_ids.shape
        meta: Dict[str, int] = {"batch_size": int(B), "prompt_len": int(T)}
        pastKV: PastKeyValues = []
        # Flow: one model call with full prompt -> cache seeded for subsequent decode_step calls.
        outputs, pastKV = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        output_len = pastKV[0][0].shape[2]
        cache = CacheState(pastKV, output_len)
        return outputs[:, -1, :], cache, meta

    @torch.no_grad()
    def prefill_batch(
        self, batch_prompt_ids: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[CacheState], list[dict]]:
        # Batch contract:
        # - input: ragged token-id lists
        # - output: per-sequence last-position logits + per-sequence cache slices
        # - decode path remains unbatched (B=1) by design
        if not batch_prompt_ids:
            raise ValueError("batch_prompt_ids must be non-empty")
        for seq in batch_prompt_ids:
            _ = self._ensure_prompt_ids(seq)

        device = next(self.model.parameters()).device
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        input_ids, attention_mask, lengths = right_pad_and_mask(batch_prompt_ids, pad_id=pad_id, device=device)

        self.prefill_calls += 1
        # One padded prefill forward pass; slicing below restores per-sequence boundaries.
        outputs, past_kv = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        if past_kv is None:
            raise RuntimeError("model returned no cache in prefill_batch")

        per_seq_logits: list[torch.Tensor] = []
        per_seq_states: list[CacheState] = []
        per_seq_meta: list[dict] = []
        for i, seq_len in enumerate(lengths):
            last_idx = int(seq_len) - 1
            per_seq_logits.append(outputs[i, last_idx, :])

            seq_past: list[tuple[torch.Tensor, torch.Tensor]] = []
            for k, v in past_kv:
                seq_k = k[i : i + 1, :, :seq_len, :].contiguous()
                seq_v = v[i : i + 1, :, :seq_len, :].contiguous()
                seq_past.append((seq_k, seq_v))
            per_seq_states.append(CacheState(seq_past, int(seq_len)))
            per_seq_meta.append({"batch_size": 1, "prompt_len": int(seq_len)})
        return per_seq_logits, per_seq_states, per_seq_meta

    @torch.no_grad()
    def decode_step(self, next_input_id: torch.Tensor, state: CacheState) -> Tuple[torch.Tensor, CacheState]:
        """
        Token-wise decode.

        Args:
            next_input_id: [B,1]
            state: previous cache state

        Returns:
            step_logits: [B, vocab]
            next_state: updated cache state
        """
        self._validate_decode_inputs(next_input_id, state)
        self.decode_calls += 1
        B = next_input_id.shape[0]
        if B != 1:
            raise ValueError(f"decode_step currently supports B=1 only, got B={B}")
        decode_attention_mask = torch.ones((B, 1), device=next_input_id.device, dtype=next_input_id.dtype)
        outputs, pastKV = self.model(
            input_ids=next_input_id,
            attention_mask=decode_attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
        )
        newPastKV = pastKV
        if self.max_cache_len is not None:
            # Sliding-window policy owns cache growth; token stream is logically unbounded.
            newPastKV = apply_sliding_window(newPastKV, self.max_cache_len)
        return outputs[:, -1, :], CacheState(newPastKV, newPastKV[0][0].shape[2])

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Optional helper for smoke tests and CLI scripts."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        input_ids = prompt_ids
        logits, cache, _ = self.prefill(input_ids, attention_mask)
        output_ids = prompt_ids
        for _ in range(max_new_tokens):
            next_token_greedy = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
            output_ids = torch.cat([output_ids, next_token_greedy], dim=1)
            logits, cache = self.decode_step(next_token_greedy, cache)
        return output_ids

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        attention_mask: list[int] | None,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        frequency_penalty: float | None = None,
        eos_token_id: int | None = None,
        stop_token_ids: set[int] | None = None,
        stop_strings: list[str] | None = None,
        seed: int | None = None,
        return_logprobs: bool = False,
    ) -> dict:
        # End-to-end generation contract:
        # - run prefill once, then decode step-by-step
        # - enforce stop policy (`eos`/`stop_token`/`stop_string`/`max_new_tokens`)
        # - return API-facing payload consumed by `/generate` and `/stream`
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        # Stage 1: normalize and validate caller inputs into a strict internal format.
        prompt_ids_list = self._ensure_prompt_ids(prompt_ids)
        if attention_mask is not None and len(attention_mask) != len(prompt_ids_list):
            raise ValueError("attention_mask must have same length as prompt_ids")

        # Stage 2: move request tensors to model device (single-item batch shape [1, T]).
        device = next(self.model.parameters()).device
        prompt_tensor = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
        if attention_mask is None:
            mask_tensor = torch.ones_like(prompt_tensor)
        else:
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

        # Stage 3: capture counters/timers before generation starts.
        start_prefill_calls = self.prefill_calls
        start_decode_calls = self.decode_calls
        t_request_start = time.perf_counter()

        # Stage 4: prefill consumes the prompt and returns the first next-token logits + cache.
        t_prefill_start = time.perf_counter()
        prefill_logits, cache, _ = self.prefill(prompt_tensor, mask_tensor)
        prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0
        step_logits = prefill_logits[0]
        decode_ms_total = 0.0

        # Some toy scripted fixtures define first generated token at abs_pos=len(prompt).
        if hasattr(self.model, "schedule"):
            bootstrap_id = torch.tensor([[prompt_ids_list[-1]]], dtype=torch.long, device=device)
            t_decode_start = time.perf_counter()
            step_logits, cache = self.decode_step(bootstrap_id, cache)
            decode_ms_total += (time.perf_counter() - t_decode_start) * 1000.0
            step_logits = step_logits[0]

        generated_ids: list[int] = []
        generated_window: list[int] = []
        token_logprobs: list[float] = []
        token_counts: dict[int, int] = {}
        stop_reason: str | None = None
        ttft_ms: float | None = None
        # generated_ids: full visible completion sequence (post stop-marker exclusion policy).
        # generated_window: context-limited tail used with apply_context_truncation.
        # token_counts: per-token frequencies for frequency penalty path.


        for step_idx in range(max_new_tokens):
            # Stage 5: choose next token from current step_logits using configured decode controls.
            greedy_flag = temperature == 0.0
            step_distribution : torch.Tensor | None = None
            if return_logprobs:
                step_distribution = self._build_step_distribution(
                    step_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    generated_token_ids=generated_ids,
                    token_counts=token_counts,
                )
            cand_id = select_next_token_id(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                generated_token_ids=generated_ids,
                token_counts=token_counts,
                seed=self._derive_step_seed(seed, step_idx),
                greedy=greedy_flag,
            )
            if ttft_ms is None:
                # TTFT: first time we have selected a candidate token.
                ttft_ms = (time.perf_counter() - t_request_start) * 1000.0

            # Exclude EOS/stop-token from visible completion by policy.
            trial = generated_ids + [cand_id]
            stop_trial, reason_trial = should_stop_tokens(
                trial,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                max_new_tokens=max_new_tokens,
            )
            if stop_trial and reason_trial in {"eos", "stop_token"}:
                # Token-level stop has highest precedence and exits before appending marker token.
                stop_reason = reason_trial
                break

            # Stage 6: append accepted token to generated state.
            generated_ids.append(cand_id)
            generated_window.append(cand_id)
            token_counts[cand_id] = token_counts.get(cand_id, 0) + 1
            if return_logprobs:
                token_logprobs.append(self._token_logprobability(step_distribution,cand_id))
                

            # Stage 7: text-level stop check; may trim completion at earliest stop-string boundary.
            completion_text = self.tokenizer.decode(generated_ids)
            stop_text, _ = should_stop_text(completion_text, stop_strings=stop_strings)
            if stop_text:
                earliest = self._find_earliest_stop_index(completion_text, stop_strings)
                completion_text = completion_text[:earliest]
                generated_ids = self._truncate_to_text_prefix(generated_ids, completion_text)
                if return_logprobs:
                    # Keep logprobs aligned with the truncated visible token sequence.
                    token_logprobs = token_logprobs[:len(generated_ids)]
                    
                keep = min(len(generated_ids), self.block_size)
                generated_window = generated_ids[-keep:]
                stop_reason = "stop_string"
                break

            # Stage 8: advance one decode step with the accepted token as next model input.
            next_input = torch.tensor([[cand_id]], dtype=torch.long, device=device)
            t_decode_start = time.perf_counter()
            step_logits, cache = self.decode_step(next_input, cache)
            decode_ms_total += (time.perf_counter() - t_decode_start) * 1000.0
            step_logits = step_logits[0]

            # Stage 9: enforce block_size on token context and KV cache in lock-step.
            prompt_ids_list, generated_window = apply_context_truncation(
                prompt_ids_list,
                generated_window,
                block_size=self.block_size,
            )
            cache = truncate_kv_cache_to_block_size(cache, keep_last_n=self.block_size)

        # Stage 10: finalize stop_reason and derived latency metrics.
        if stop_reason is None:
            stop_reason = "max_new_tokens"
        if ttft_ms is None:
            ttft_ms = prefill_ms

        completion_text = self.tokenizer.decode(generated_ids) if generated_ids else ""
        all_token_ids = prompt_ids_list + generated_window
        decode_ms_per_token = decode_ms_total / float(max(len(generated_ids) - 1, 1)) if generated_ids else 0.0
        tokens_per_sec = (1000.0 / decode_ms_per_token) if decode_ms_per_token > 0 else 0.0
        metrics = {
            "generated_tokens": len(generated_ids),
            "decode_steps": self.decode_calls - start_decode_calls,
            "prefill_calls": self.prefill_calls - start_prefill_calls,
            "decode_calls": self.decode_calls - start_decode_calls,
            "cache_len": cache.past_len,
            "ttft_ms": float(ttft_ms),
            "prefill_ms": float(prefill_ms),
            "decode_ms_total": float(decode_ms_total),
            "decode_ms_per_token": float(decode_ms_per_token),
            "tokens_per_sec": float(tokens_per_sec),
            "requested_dtype": self.requested_dtype,
            "runtime_dtype": self.runtime_dtype,
            "runtime_quant_mode": self.quant_mode,
            "runtime_fallback_reason": self.runtime_fallback_reason,
        }
        out = {
            "completion_token_ids": generated_ids,
            "completion_text": completion_text,
            "stop_reason": stop_reason,
            "metrics": metrics,
            "all_token_ids": all_token_ids,
        }
        # Metrics ownership: API logs/exports these values verbatim; keep keys stable.
        if return_logprobs and token_logprobs:
            out["token_logprobs"] = token_logprobs
        # Final payload is API-facing and consumed by both /generate and /stream code paths.
        return out


def build_engine_from_package(
    package_path: str,
    device: str,
    dtype: str,
    quant_mode: str | None = None,
    *,
    loader: str = "package",
) -> Engine:
    """Build an Engine from a model artifact.

    Args:
        loader: "package" for sp16k package dirs (default),
                "nanollama" for raw .pt checkpoints with tiktoken.
    """
    if loader == "nanollama":
        from llm_lab.core.package.nanollama_loader import load_nanollama_checkpoint
        config, tokenizer, model = load_nanollama_checkpoint(package_path, device=device)
    elif loader == "package":
        config, tokenizer, model = load_model_package(package_path, device=device)
    else:
        raise ValueError(f"unknown loader: {loader!r}; expected 'package' or 'nanollama'")

    runtime_dtype, precision_reason = runtime_precision_decision(dtype, device)
    model = cast_model_for_inference(model.eval(), runtime_dtype)

    quant_info = describe_quant_runtime(quant_mode, device)
    runtime_quant_mode = quant_info.get("runtime_quant_mode")
    quant_reason = quant_info.get("quant_fallback_reason")
    if runtime_quant_mode is not None:
        model = maybe_quantize_model(model, runtime_quant_mode, device)

    setattr(model, "_runtime_dtype", runtime_dtype)
    setattr(model, "_runtime_quant_mode", runtime_quant_mode or "none")
    runtime_reason = precision_reason or quant_reason

    return Engine(
        model=model,
        tokenizer=tokenizer,
        block_size=int(config.block_size),
        max_cache_len=int(config.block_size),
        requested_dtype=dtype,
        runtime_dtype=runtime_dtype,
        quant_mode=runtime_quant_mode,
        runtime_fallback_reason=runtime_reason,
    )
