# tests/serving/test_api_stream_sse.py
from __future__ import annotations

import json


def _base_payload() -> dict:
    return {
        "prompt": "a",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stop_strings": None,
        "stop_token_ids": None,
        "seed": 7,
        "return_logprobs": False,
    }


def _collect_sse_events(client, payload: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    with client.stream("POST", "/stream", json=payload) as r:
        assert r.status_code == 200
        current_event = "message"
        for line in r.iter_lines():
            if line is None or line == "":
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue
            if line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                out.append((current_event, data))
    return out


def test_stream_emits_token_events_then_final(client) -> None:
    events = _collect_sse_events(client, _base_payload())
    kinds = [k for k, _ in events]
    token_count = sum(1 for k in kinds if k == "token")
    final_count = sum(1 for k in kinds if k == "final")
    assert token_count >= 1
    assert final_count == 1
    assert kinds[-1] == "final"


def test_stream_final_event_contains_metrics(client) -> None:
    events = _collect_sse_events(client, _base_payload())
    final = [payload for kind, payload in events if kind == "final"][0]
    metrics = final["metrics"]
    for key in ("ttft_ms", "prefill_ms", "decode_ms_total", "decode_ms_per_token", "tokens_per_sec"):
        assert key in metrics


def test_stream_ttft_positive_and_reasonable(client) -> None:
    events = _collect_sse_events(client, _base_payload())
    final = [payload for kind, payload in events if kind == "final"][0]
    ttft = float(final["metrics"]["ttft_ms"])
    assert 0.0 < ttft < 20000.0


def test_stream_respects_stop_reason_and_excludes_markers(client) -> None:
    payload = _base_payload()
    payload["stop_strings"] = ["yz"]
    events = _collect_sse_events(client, payload)
    final = [p for kind, p in events if kind == "final"][0]
    assert final["stop_reason"] == "stop_string"
    assert "yz" not in final["completion_text"]
