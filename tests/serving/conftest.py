# tests/serving/conftest.py
from __future__ import annotations
import os 
from pathlib import Path
import logging
from io import StringIO

import pytest
import torch
from fastapi.testclient import TestClient

from llm_lab.core.package.io import load_model_package
from llm_lab.serving.api import create_app
from llm_lab.serving.config import ServingConfig
from llm_lab.serving.engine import Engine

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@pytest.fixture(scope="session")
def serving_package_dir() -> Path:
    p = os.environ.get("SERVING_PACKAGE_DIR","experiments/p1_pos_enc/runs/rope/package")
    pkg = Path(p)
    if not pkg.exists():
        pytest.skip(f"SERVING_PACKAGE_DIR does not exist: {pkg}")
    return pkg


@pytest.fixture(scope="session")
def serving_device() -> str:
    return _pick_device()

@pytest.fixture(scope="session")
def serving_pkg(serving_package_dir: Path,serving_device:str):
    config, tokenizer, model = load_model_package(serving_package_dir, device=serving_device)
    model.eval()
    return config, tokenizer, model


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 8

    def __init__(self) -> None:
        self._stoi = {
            "<|pad|>": 0,
            "a": 1,
            "b": 2,
            "c": 3,
            "x": 4,
            "y": 5,
            "z": 6,
            " ": 7,
            "<|endoftext|>": 8,
        }
        self._itos = {v: k for k, v in self._stoi.items()}

    def token_to_id(self, tok: str) -> int:
        return int(self._stoi[tok])

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for ch in text:
            if ch not in self._stoi:
                raise KeyError(ch)
            out.append(self._stoi[ch])
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids if self._itos[i] not in {"<|pad|>", "<|endoftext|>"})


class ToyScheduleModel(torch.nn.Module):
    """Deterministic logits by absolute position for serving API tests."""

    def __init__(self, vocab_size: int = 9, schedule: dict[int, int] | None = None, default_id: int = 1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.schedule = {} if schedule is None else dict(schedule)
        self.default_id = int(default_id)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def _pick(self, abs_pos: int) -> int:
        return int(self.schedule.get(abs_pos, self.default_id))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        logits = torch.full((B, T, self.vocab_size), -10.0, device=device)
        past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
        for t in range(T):
            pick = self._pick(past_len + t)
            logits[:, t, pick] = 10.0
        if not use_cache:
            return logits, None
        total = past_len + T
        k = torch.zeros(B, 1, total, 1, device=device)
        v = torch.zeros(B, 1, total, 1, device=device)
        return logits, [(k, v)]


@pytest.fixture()
def toy_engine() -> Engine:
    tok = ToyTokenizer()
    # First generated token after prompt of len=1 is x, then y, then z.
    model = ToyScheduleModel(schedule={1: 4, 2: 5, 3: 6, 4: 2, 5: 3}, default_id=1)
    return Engine(model=model, tokenizer=tok, block_size=16, max_cache_len=16)


@pytest.fixture()
def app_factory(toy_engine: Engine):
    def _factory(
        *,
        rate_limit_max_requests: int = 100,
        rate_limit_window_s: float = 60.0,
        clock=None,
        log_raw_prompts: bool = False,
    ):
        cfg = ServingConfig(
            rate_limit_enabled=True,
            rate_limit_max_requests=rate_limit_max_requests,
            rate_limit_window_s=rate_limit_window_s,
            request_clock=clock,
            log_raw_prompts=log_raw_prompts,
        )
        return create_app(toy_engine, config=cfg)

    return _factory


@pytest.fixture()
def client(app_factory):
    app = app_factory()
    with TestClient(app) as tc:
        yield tc


@pytest.fixture()
def metrics_store(client):
    return client.app.state.metrics_store


class FakeClock:
    def __init__(self, t0: float = 1000.0) -> None:
        self._t = float(t0)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += float(dt)


@pytest.fixture()
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def capture_logs():
    logger = logging.getLogger("llm_lab.serving.requests")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
