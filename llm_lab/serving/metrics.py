# llm_lab/serving/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class RequestMetrics:
    request_id: str
    endpoint: str
    status_code: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    prefill_ms: float
    decode_ms_total: float
    decode_ms_per_token: float
    tokens_per_sec: float
    stop_reason: str | None = None
    error_category: str | None = None


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._error_count = 0
        self._rate_limited_count = 0

    def observe(self, m: RequestMetrics) -> None:
        with self._lock:
            self._request_count += 1
            self._prompt_tokens_total += int(m.prompt_tokens)
            self._completion_tokens_total += int(m.completion_tokens)
            if m.status_code >= 400:
                self._error_count += 1
            if m.status_code == 429:
                self._rate_limited_count += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self._request_count,
                "prompt_tokens_total": self._prompt_tokens_total,
                "completion_tokens_total": self._completion_tokens_total,
                "errors_total": self._error_count,
                "rate_limited_total": self._rate_limited_count,
            }

    def render_text(self) -> str:
        s = self.snapshot()
        lines = [
            f"requests_total {s['requests_total']}",
            f"prompt_tokens_total {s['prompt_tokens_total']}",
            f"completion_tokens_total {s['completion_tokens_total']}",
            f"errors_total {s['errors_total']}",
            f"rate_limited_total {s['rate_limited_total']}",
        ]
        return "\n".join(lines) + "\n"
