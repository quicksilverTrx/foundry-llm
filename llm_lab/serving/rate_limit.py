# llm_lab/serving/rate_limit.py
from __future__ import annotations

from collections import deque
from typing import Callable


class RateLimiter:
    def __init__(self, *, max_requests: int, window_s: float, clock: Callable[[], float] | None = None) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.max_requests = int(max_requests)
        self.window_s = float(window_s)
        self.clock = clock
        self._events: dict[str, deque[float]] = {}

    def _now(self) -> float:
        if self.clock is not None:
            return float(self.clock())
        import time

        return time.time()

    def _prune(self, key: str, now: float) -> deque[float]:
        q = self._events.setdefault(key, deque())
        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
        return q

    def allow(self, key: str) -> bool:
        now = self._now()
        q = self._prune(key, now)
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True

    def retry_after_seconds(self, key: str) -> float:
        now = self._now()
        q = self._prune(key, now)
        if not q:
            return 0.0
        return max(0.0, self.window_s - (now - q[0]))
