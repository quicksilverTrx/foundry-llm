# llm_lab/serving/logging.py
from __future__ import annotations

import json
import logging
from typing import Any

from llm_lab.serving._shared import sha256_text
from llm_lab.serving.metrics import RequestMetrics


LOGGER = logging.getLogger("llm_lab.serving.requests")


def build_privacy_log_record(
    *,
    request_id: str,
    req: Any,
    resp_meta: dict,
    metrics: RequestMetrics,
    config: Any,
) -> dict:
    prompt = str(getattr(req, "prompt", ""))
    completion_text = str(resp_meta.get("completion_text", ""))
    prompt_tokens = int(resp_meta.get("prompt_tokens", 0))
    completion_tokens = int(resp_meta.get("completion_tokens", 0))

    record = {
        "request_id": request_id,
        "endpoint": metrics.endpoint,
        "status_code": metrics.status_code,
        "prompt_len": len(prompt),
        "prompt_token_count": prompt_tokens,
        "prompt_hash": sha256_text(prompt),
        "completion_len": len(completion_text),
        "completion_token_count": completion_tokens,
        "ttft_ms": float(metrics.ttft_ms),
        "prefill_ms": float(metrics.prefill_ms),
        "decode_ms_total": float(metrics.decode_ms_total),
        "decode_ms_per_token": float(metrics.decode_ms_per_token),
        "tokens_per_sec": float(metrics.tokens_per_sec),
        "stop_reason": metrics.stop_reason,
        "error_category": metrics.error_category,
    }

    if bool(getattr(config, "log_raw_prompts", False)):
        record["prompt"] = prompt
        record["completion_text"] = completion_text

    return record


def log_request(record: dict) -> None:
    LOGGER.info(json.dumps(record, sort_keys=True))
