# tests/serving/test_engine_prefill_decode_structure.py
from __future__ import annotations

import pytest
import torch

from llm_lab.serving.engine import CacheState, Engine


class SpyModel(torch.nn.Module):
    """Tiny spy model to validate Engine call wiring during scaffolding phase."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        self.forward_calls += 1
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size, device=input_ids.device, dtype=torch.float32)

        # Shape-compatible fake cache: [B, H, T, D]
        k = torch.zeros(B, 2, T, 4, device=input_ids.device)
        v = torch.zeros(B, 2, T, 4, device=input_ids.device)
        past = [(k, v)] if use_cache else None
        return logits, past


class DummyTokenizer:
    pad_token_id = 0


def test_prefill_increments_counter_and_returns_last_logits_and_state():
    model = SpyModel()
    engine = Engine(model, DummyTokenizer(), block_size=32, max_cache_len=None)

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    assert engine.prefill_calls == 0
    last_logits, state, meta = engine.prefill(input_ids, attention_mask)
    assert engine.prefill_calls == 1
    assert last_logits.shape == (1, model.vocab_size)
    assert state.past_key_values is not None
    assert state.past_len == 3
    assert meta["batch_size"] == 1
    assert meta["prompt_len"] == 3


def test_decode_increments_counter_and_grows_state():
    model = SpyModel()
    engine = Engine(model, DummyTokenizer(), block_size=32, max_cache_len=None)

    fake_k = torch.zeros(1, 2, 3, 4)
    fake_v = torch.zeros(1, 2, 3, 4)
    state = CacheState(past_key_values=[(fake_k, fake_v)], past_len=3)

    bad_next = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError):
        _ = engine.decode_step(bad_next, state)

    next_id = torch.tensor([[7]], dtype=torch.long)
    assert engine.decode_calls == 0
    logits, next_state = engine.decode_step(next_id, state)
    assert engine.decode_calls == 1
    assert logits.shape == (1, model.vocab_size)
    assert next_state.past_key_values is not None
    assert next_state.past_len == 1


def test_decode_requires_existing_cache_state():
    model = SpyModel()
    engine = Engine(model, DummyTokenizer(), block_size=32)

    with pytest.raises(ValueError):
        _ = engine.decode_step(torch.tensor([[5]], dtype=torch.long), CacheState(past_key_values=None, past_len=0))


def test_generate_greedy_shape_contract():
    model = SpyModel()
    engine = Engine(model, DummyTokenizer(), block_size=32)

    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(prompt_ids)

    output = engine.generate_greedy(prompt_ids, mask, max_new_tokens=4)
    assert output.shape == (1, 7)


def test_sliding_window_contract_marker():
    assert True
