# llm_lab/serving/__init__.py
"""Serving inference modules."""

from llm_lab.serving.batching import right_pad_and_mask
from llm_lab.serving.config import ServingConfig
from llm_lab.serving.decode_controls import (
    apply_context_truncation,
    should_stop_text,
    should_stop_tokens,
    truncate_kv_cache_to_block_size,
)
from llm_lab.serving.engine import CacheState, Engine, build_engine_from_package
from llm_lab.serving.kv_cache import TIME_DIM, PastKV, PastKVs, apply_sliding_window, kv_append, kv_truncate
from llm_lab.serving.metrics import MetricsStore, RequestMetrics
from llm_lab.serving.precision import cast_model_for_inference, resolve_runtime_precision, validate_precision_request
from llm_lab.serving.quant import describe_quant_runtime, load_quantized_model_package, maybe_quantize_model, quant_backend_is_available
from llm_lab.serving.rate_limit import RateLimiter
from llm_lab.serving.sampling import (
    apply_frequency_penalty,
    apply_repetition_penalty,
    apply_temperature,
    sample_next_token_id,
    select_next_token_id,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "CacheState",
    "Engine",
    "build_engine_from_package",
    "ServingConfig",
    "RateLimiter",
    "RequestMetrics",
    "MetricsStore",
    "PastKV",
    "PastKVs",
    "TIME_DIM",
    "kv_append",
    "kv_truncate",
    "apply_sliding_window",
    "right_pad_and_mask",
    "should_stop_tokens",
    "should_stop_text",
    "apply_context_truncation",
    "truncate_kv_cache_to_block_size",
    "apply_temperature",
    "top_k_filter",
    "top_p_filter",
    "apply_repetition_penalty",
    "apply_frequency_penalty",
    "sample_next_token_id",
    "select_next_token_id",
    "resolve_runtime_precision",
    "cast_model_for_inference",
    "validate_precision_request",
    "quant_backend_is_available",
    "maybe_quantize_model",
    "load_quantized_model_package",
    "describe_quant_runtime",
]
