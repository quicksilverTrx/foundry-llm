# tests/serving/test_api_generate.py
from __future__ import annotations


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


def test_health_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"


def test_generate_smoke_returns_required_fields(client) -> None:
    r = client.post("/generate", json=_base_payload())
    assert r.status_code == 200
    out = r.json()
    for key in ("request_id", "completion_text", "completion_token_ids", "stop_reason", "metrics"):
        assert key in out


def test_generate_respects_max_new_tokens(client) -> None:
    payload = _base_payload()
    payload["max_new_tokens"] = 2
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert len(out["completion_token_ids"]) <= 2


def test_generate_stop_markers_excluded(client) -> None:
    payload = _base_payload()
    payload["stop_strings"] = ["yz"]
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["stop_reason"] == "stop_string"
    assert "yz" not in out["completion_text"]


def test_generate_validates_bad_inputs_422(client) -> None:
    bad = _base_payload()
    bad["max_new_tokens"] = -1
    r = client.post("/generate", json=bad)
    assert r.status_code == 422

    bad2 = _base_payload()
    bad2["top_p"] = 1.5
    r2 = client.post("/generate", json=bad2)
    assert r2.status_code == 422
