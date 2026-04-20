# tests/serving/test_rate_limit.py
from __future__ import annotations

from fastapi.testclient import TestClient


def _payload() -> dict:
    return {
        "prompt": "a",
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "stop_strings": None,
        "stop_token_ids": None,
        "seed": 0,
        "return_logprobs": False,
    }


def test_rate_limit_returns_429_after_threshold(app_factory) -> None:
    app = app_factory(rate_limit_max_requests=2, rate_limit_window_s=10.0)
    with TestClient(app) as client:
        headers = {"X-Client-Key": "same-user"}
        r1 = client.post("/generate", json=_payload(), headers=headers)
        r2 = client.post("/generate", json=_payload(), headers=headers)
        r3 = client.post("/generate", json=_payload(), headers=headers)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        assert r3.json()["error"]["code"] == "rate_limited"


def test_rate_limit_resets_after_window(app_factory, fake_clock) -> None:
    app = app_factory(rate_limit_max_requests=2, rate_limit_window_s=5.0, clock=fake_clock.now)
    with TestClient(app) as client:
        headers = {"X-Client-Key": "clock-user"}
        assert client.post("/generate", json=_payload(), headers=headers).status_code == 200
        assert client.post("/generate", json=_payload(), headers=headers).status_code == 200
        assert client.post("/generate", json=_payload(), headers=headers).status_code == 429

        fake_clock.advance(6.0)
        assert client.post("/generate", json=_payload(), headers=headers).status_code == 200
