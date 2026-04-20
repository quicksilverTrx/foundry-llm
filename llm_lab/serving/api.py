# llm_lab/serving/api.py
from __future__ import annotations

import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from llm_lab.serving._shared import resolve_eos_token_id
from llm_lab.serving.config import ServingConfig
from llm_lab.serving.logging import build_privacy_log_record, log_request
from llm_lab.serving.metrics import MetricsStore, RequestMetrics
from llm_lab.serving.rate_limit import RateLimiter
from llm_lab.serving.safety import build_refusal_text, postprocess_generated_text, should_refuse_prompt
from llm_lab.serving.schemas import GenerateRequest, GenerateResponse
from llm_lab.serving.stream import iter_token_sse, sse_encode, StreamRequest

if TYPE_CHECKING:
    from llm_lab.serving.engine import Engine


def _client_key(request: Request, *, config: ServingConfig) -> str:
    # Conservative identity derivation for rate limiting:
    # custom key header -> x-forwarded-for first hop -> direct client host.
    header_name = config.rate_limit_key_header.lower()
    from_header = request.headers.get(header_name)
    if from_header:
        return _normalize_key(from_header)

    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first_hop = fwd.split(",")[0].strip()
        if first_hop:
            return _normalize_key(first_hop)

    if request.client is not None and request.client.host:
        return _normalize_key(request.client.host)
    return "unknown"


def _normalize_key(raw: str) -> str:
    x = raw.strip()
    if x == "":
        return "unknown"
    return x[:128]


def _is_valid_request_id(value: str) -> bool:
    # Keep validation simple and explicit for API safety.
    if len(value) < 8 or len(value) > 128:
        return False
    return re.fullmatch(r"[A-Za-z0-9._:-]+", value) is not None


def _resolve_request_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id")
    if incoming and _is_valid_request_id(incoming):
        return incoming
    return str(uuid.uuid4())


def _map_exception(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return 400, "bad_request", "validation"
    if isinstance(exc, NotImplementedError):
        return 501, "not_implemented", "not_implemented"
    return 500, "internal_error", "internal"


def _parse_final_sse_chunk(chunk: str) -> dict | None:
    if chunk is None:
        return None
    if "event: final" not in chunk:
        return None
    for line in chunk.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            return json.loads(payload)
    return None


def _zero_metrics_payload(engine: "Engine") -> dict[str, Any]:
    return {
        "generated_tokens": 0,
        "decode_steps": 0,
        "prefill_calls": 0,
        "decode_calls": 0,
        "cache_len": 0,
        "ttft_ms": 0.0,
        "prefill_ms": 0.0,
        "decode_ms_total": 0.0,
        "decode_ms_per_token": 0.0,
        "tokens_per_sec": 0.0,
        "requested_dtype": getattr(engine, "requested_dtype", "fp32"),
        "runtime_dtype": getattr(engine, "runtime_dtype", "fp32"),
        "runtime_quant_mode": getattr(engine, "quant_mode", None),
        "runtime_fallback_reason": getattr(engine, "runtime_fallback_reason", None),
    }


def create_app(engine: "Engine", *, config: ServingConfig) -> FastAPI:
    app = FastAPI(title="foundry-llm serving", version="0.3.0")

    # Flow note: app-level stores are shared by all handlers.
    # - metrics_store: append-only request aggregates for `/metrics`.
    # - limiter: optional fixed-window gate for `/generate` + `/stream`.
    metrics_store = MetricsStore()
    limiter = None
    if config.rate_limit_enabled:
        limiter = RateLimiter(
            max_requests=config.rate_limit_max_requests,
            window_s=config.rate_limit_window_s,
            clock=config.request_clock,
        )

    app.state.metrics_store = metrics_store
    app.state.rate_limiter = limiter

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "block_size": int(engine.block_size),
            "max_cache_len": int(engine.max_cache_len) if engine.max_cache_len is not None else None,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return metrics_store.render_text()

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest, request: Request):
        # Request lifecycle (sync path):
        # 1) derive request_id + client key
        # 2) rate-limit check
        # 3) run engine.generate once
        # 4) map output -> API schema
        # 5) observe metrics + emit privacy log record
        request_id = _resolve_request_id(request)
        key = _client_key(request, config=config)
        if limiter is not None and not limiter.allow(key):
            # Contract: 429 responses are still fully observed in metrics/logging.
            retry_after = limiter.retry_after_seconds(key)
            m = RequestMetrics(
                request_id=request_id,
                endpoint="/generate",
                status_code=429,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms_total=0.0,
                decode_ms_per_token=0.0,
                tokens_per_sec=0.0,
                stop_reason="rate_limited",
                error_category="rate_limit",
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={"completion_text": "", "prompt_tokens": 0, "completion_tokens": 0},
                metrics=m,
                config=config,
            )
            log_request(rec)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after)))},
                content={"error": {"code": "rate_limited", "message": "rate limit exceeded"}},
            )

        # Safety pre-filter: short-circuit generation on sensitive prompt patterns.
        prompt_refused, prompt_reason_codes = should_refuse_prompt(req.prompt)
        if prompt_refused:
            refusal_text = build_refusal_text(prompt_reason_codes)
            try:
                prompt_token_count = len(engine.tokenizer.encode(req.prompt))
            except Exception:
                prompt_token_count = len(req.prompt)
            payload = {
                "request_id": request_id,
                "completion_text": refusal_text,
                "completion_token_ids": [],
                "stop_reason": "safety_refusal",
                "metrics": _zero_metrics_payload(engine),
                "token_logprobs": None,
                "safety_flags": prompt_reason_codes,
                "refusal_applied": True,
            }
            m = RequestMetrics(
                request_id=request_id,
                endpoint="/generate",
                status_code=200,
                prompt_tokens=prompt_token_count,
                completion_tokens=0,
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms_total=0.0,
                decode_ms_per_token=0.0,
                tokens_per_sec=0.0,
                stop_reason="safety_refusal",
                error_category=None,
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={"completion_text": refusal_text, "prompt_tokens": prompt_token_count, "completion_tokens": 0},
                metrics=m,
                config=config,
            )
            log_request(rec)
            return payload

        try:
            # Data contract: prompt is tokenized once in API layer,
            # then engine owns decode loop + stop-policy accounting.
            prompt_ids = engine.tokenizer.encode(req.prompt)
            out = engine.generate(
                prompt_ids=prompt_ids,
                attention_mask=[1] * len(prompt_ids),
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                eos_token_id=resolve_eos_token_id(engine.tokenizer),
                stop_token_ids=set(req.stop_token_ids) if req.stop_token_ids else None,
                stop_strings=req.stop_strings,
                seed=req.seed,
                return_logprobs=req.return_logprobs,
            )
        except Exception as exc:
            http_status, public_code, category = _map_exception(exc)
            m = RequestMetrics(
                request_id=request_id,
                endpoint="/generate",
                status_code=http_status,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms_total=0.0,
                decode_ms_per_token=0.0,
                tokens_per_sec=0.0,
                stop_reason="error",
                error_category=category,
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={"completion_text": "", "prompt_tokens": 0, "completion_tokens": 0},
                metrics=m,
                config=config,
            )
            log_request(rec)
            return JSONResponse(status_code=http_status, content={"error": {"code": public_code, "message": str(exc)}})

        safe_completion, post_flags, post_refusal = postprocess_generated_text(str(out["completion_text"]))
        completion_token_ids = list(out["completion_token_ids"])
        stop_reason = out["stop_reason"]
        token_logprobs = out.get("token_logprobs")
        if post_refusal:
            completion_token_ids = []
            stop_reason = "safety_refusal"
            token_logprobs = None

        payload = {
            "request_id": request_id,
            "completion_text": safe_completion,
            "completion_token_ids": completion_token_ids,
            "stop_reason": stop_reason,
            "metrics": out["metrics"],
            "token_logprobs": token_logprobs,
            "safety_flags": post_flags,
            "refusal_applied": post_refusal,
        }
        # Ownership: API assembles client-visible shape; engine provides stop_reason/metrics values.

        m = RequestMetrics(
            request_id=request_id,
            endpoint="/generate",
            status_code=200,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_token_ids),
            ttft_ms=float(out["metrics"]["ttft_ms"]),
            prefill_ms=float(out["metrics"]["prefill_ms"]),
            decode_ms_total=float(out["metrics"]["decode_ms_total"]),
            decode_ms_per_token=float(out["metrics"]["decode_ms_per_token"]),
            tokens_per_sec=float(out["metrics"]["tokens_per_sec"]),
            stop_reason=stop_reason,
            error_category=None,
        )
        metrics_store.observe(m)
        rec = build_privacy_log_record(
            request_id=request_id,
            req=req,
            resp_meta={
                "completion_text": safe_completion,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_token_ids),
            },
            metrics=m,
            config=config,
        )
        log_request(rec)
        return payload

    @app.post("/stream")
    def stream(req: StreamRequest, request: Request):
        # Stream lifecycle (generator path):
        # 1) gate on rate limit
        # 2) run token SSE iterator
        # 3) capture final payload (if present) or synthesize fallback
        # 4) persist metrics/log record once on termination path
        request_id = _resolve_request_id(request)
        key = _client_key(request, config=config)

        if limiter is not None and not limiter.allow(key):
            retry_after = limiter.retry_after_seconds(key)
            m = RequestMetrics(
                request_id=request_id,
                endpoint="/stream",
                status_code=429,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms_total=0.0,
                decode_ms_per_token=0.0,
                tokens_per_sec=0.0,
                stop_reason="rate_limited",
                error_category="rate_limit",
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={"completion_text": "", "prompt_tokens": 0, "completion_tokens": 0},
                metrics=m,
                config=config,
            )
            log_request(rec)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after)))},
                content={"error": {"code": "rate_limited", "message": "rate limit exceeded"}},
            )

        try:
            prompt_tokens = len(engine.tokenizer.encode(req.prompt))
        except Exception as exc:
            http_status, public_code, category = _map_exception(exc)
            m = RequestMetrics(
                request_id=request_id,
                endpoint="/stream",
                status_code=http_status,
                prompt_tokens=0,
                completion_tokens=0,
                ttft_ms=0.0,
                prefill_ms=0.0,
                decode_ms_total=0.0,
                decode_ms_per_token=0.0,
                tokens_per_sec=0.0,
                stop_reason="error",
                error_category=category,
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={"completion_text": "", "prompt_tokens": 0, "completion_tokens": 0},
                metrics=m,
                config=config,
            )
            log_request(rec)
            return JSONResponse(status_code=http_status, content={"error": {"code": public_code, "message": str(exc)}})

        prompt_refused, prompt_reason_codes = should_refuse_prompt(req.prompt)
        if prompt_refused:
            refusal_text = build_refusal_text(prompt_reason_codes)

            def _refusal_iterator():
                final_payload = {
                    "stop_reason": "safety_refusal",
                    "completion_text": refusal_text,
                    "completion_token_ids": [],
                    "metrics": _zero_metrics_payload(engine),
                    "safety_flags": prompt_reason_codes,
                    "refusal_applied": True,
                }
                yield sse_encode("final", final_payload)

                m = RequestMetrics(
                    request_id=request_id,
                    endpoint="/stream",
                    status_code=200,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    ttft_ms=0.0,
                    prefill_ms=0.0,
                    decode_ms_total=0.0,
                    decode_ms_per_token=0.0,
                    tokens_per_sec=0.0,
                    stop_reason="safety_refusal",
                    error_category=None,
                )
                metrics_store.observe(m)
                rec = build_privacy_log_record(
                    request_id=request_id,
                    req=req,
                    resp_meta={
                        "completion_text": refusal_text,
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": 0,
                    },
                    metrics=m,
                    config=config,
                )
                log_request(rec)

            return StreamingResponse(_refusal_iterator(), media_type="text/event-stream")
        started = time.perf_counter()

        def _iterator():
            final_payload: dict | None = None
            interrupted = False
            stream_error: Exception | None = None
            # Data contract: `final_payload` is the single source for completion metrics
            # written to MetricsStore/logs after iteration ends.
            try:
                for chunk in iter_token_sse(engine=engine, req=req):
                    # We parse `event: final` back out of the stream so API-layer
                    # observability stays aligned with exactly what the client receives.
                    maybe_final = _parse_final_sse_chunk(chunk)
                    if maybe_final is not None:
                        safe_completion, post_flags, post_refusal = postprocess_generated_text(
                            str(maybe_final.get("completion_text", ""))
                        )
                        maybe_final["completion_text"] = safe_completion
                        maybe_final["safety_flags"] = post_flags
                        maybe_final["refusal_applied"] = post_refusal
                        if post_refusal:
                            maybe_final["stop_reason"] = "safety_refusal"
                            maybe_final["completion_token_ids"] = []
                            maybe_final.pop("token_logprobs", None)
                        chunk = sse_encode("final", maybe_final)
                        final_payload = maybe_final
                    yield chunk
            except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                interrupted = True
                return
            except Exception as exc:
                stream_error = exc

            if final_payload is None:
                stop_reason = "cancelled" if interrupted else "error"
                final_payload = {
                    "stop_reason": stop_reason,
                    "completion_text": "",
                    "completion_token_ids": [],
                    "metrics": {
                        "ttft_ms": (time.perf_counter() - started) * 1000.0,
                        "prefill_ms": 0.0,
                        "decode_ms_total": 0.0,
                        "decode_ms_per_token": 0.0,
                        "tokens_per_sec": 0.0,
                    },
                }

            status_code = 200
            error_category = None
            if interrupted:
                status_code = 499
                error_category = "cancelled"
            elif stream_error is not None:
                mapped_status, _, mapped_category = _map_exception(stream_error)
                status_code = mapped_status
                error_category = mapped_category

            m = RequestMetrics(
                request_id=request_id,
                endpoint="/stream",
                status_code=status_code,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(final_payload.get("completion_token_ids", [])),
                ttft_ms=float(final_payload["metrics"].get("ttft_ms", 0.0)),
                prefill_ms=float(final_payload["metrics"].get("prefill_ms", 0.0)),
                decode_ms_total=float(final_payload["metrics"].get("decode_ms_total", 0.0)),
                decode_ms_per_token=float(final_payload["metrics"].get("decode_ms_per_token", 0.0)),
                tokens_per_sec=float(final_payload["metrics"].get("tokens_per_sec", 0.0)),
                stop_reason=str(final_payload.get("stop_reason")),
                error_category=error_category,
            )
            metrics_store.observe(m)
            rec = build_privacy_log_record(
                request_id=request_id,
                req=req,
                resp_meta={
                    "completion_text": str(final_payload.get("completion_text", "")),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": len(final_payload.get("completion_token_ids", [])),
                },
                metrics=m,
                config=config,
            )
            log_request(rec)

        return StreamingResponse(_iterator(), media_type="text/event-stream")

    return app
