# tests/serving/test_metrics_and_logging.py
from __future__ import annotations

import json


def _payload(prompt: str = "a") -> dict:
    return {
        "prompt": prompt,
        "max_new_tokens": 3,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stop_strings": None,
        "stop_token_ids": None,
        "seed": 123,
        "return_logprobs": False,
    }


def _last_json_line(capture_logs) -> dict:
    lines = [x.strip() for x in capture_logs.getvalue().splitlines() if x.strip()]
    assert lines, "expected at least one log line"
    return json.loads(lines[-1])


def test_logs_do_not_contain_raw_prompt_by_default(capture_logs, client) -> None:
    marker = "xyz"
    r = client.post("/generate", json=_payload(prompt=marker))
    assert r.status_code == 200
    text = capture_logs.getvalue()
    assert marker not in text


def test_log_record_contains_required_fields(capture_logs, client) -> None:
    r = client.post("/generate", json=_payload(prompt="abc"))
    assert r.status_code == 200
    rec = _last_json_line(capture_logs)
    for key in (
        "request_id",
        "prompt_len",
        "prompt_hash",
        "ttft_ms",
        "prefill_ms",
        "decode_ms_per_token",
        "stop_reason",
    ):
        assert key in rec


def test_metrics_store_observes_requests(metrics_store, client) -> None:
    r1 = client.post("/generate", json=_payload(prompt="abc"))
    r2 = client.post("/generate", json=_payload(prompt="xyz"))
    assert r1.status_code == 200
    assert r2.status_code == 200

    snap = metrics_store.snapshot()
    assert snap["requests_total"] >= 2
    assert snap["prompt_tokens_total"] >= 6
    assert snap["completion_tokens_total"] >= 0


def test_metrics_endpoint_returns_text(client) -> None:
    _ = client.post("/generate", json=_payload(prompt="abc"))
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert body.strip() != ""
    assert "requests_total" in body
    assert "prompt_tokens_total" in body
