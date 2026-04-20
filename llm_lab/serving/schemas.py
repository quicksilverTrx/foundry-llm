# llm_lab/serving/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=0, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=10.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop_strings: list[str] | None = Field(default=None)
    stop_token_ids: list[int] | None = Field(default=None)
    seed: int | None = Field(default=None)
    return_logprobs: bool = Field(default=False)


# /stream uses the same schema as /generate.
StreamRequest = GenerateRequest


class GenerationMetrics(BaseModel):
    generated_tokens: int
    decode_steps: int
    prefill_calls: int
    decode_calls: int
    cache_len: int
    ttft_ms: float
    prefill_ms: float
    decode_ms_total: float
    decode_ms_per_token: float
    tokens_per_sec: float
    requested_dtype: str | None = None
    runtime_dtype: str | None = None
    runtime_quant_mode: str | None = None
    runtime_fallback_reason: str | None = None


class GenerateResponse(BaseModel):
    request_id: str
    completion_text: str
    completion_token_ids: list[int]
    stop_reason: str
    metrics: GenerationMetrics
    token_logprobs: list[float] | None = None
    safety_flags: list[str] | None = None
    refusal_applied: bool = False
