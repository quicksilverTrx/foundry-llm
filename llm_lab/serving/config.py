# llm_lab/serving/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cpu"
    dtype: str = "fp32"
    quant_mode: str | None = None
    log_raw_prompts: bool = False
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 60
    rate_limit_window_s: float = 60.0
    rate_limit_key_header: str = "x-client-key"
    request_clock: Callable[[], float] | None = None
