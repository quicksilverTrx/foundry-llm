# tests/serving/test_stop_eos_truncation.py
from __future__ import annotations

import torch

from llm_lab.serving.decode_controls import (
    apply_context_truncation,
    should_stop_text,
    truncate_kv_cache_to_block_size,
)
from llm_lab.serving.engine import CacheState, Engine


class PieceTokenizer:
    def __init__(self, id_to_piece: dict[int, str]):
        self.id_to_piece = dict(id_to_piece)
        self.piece_to_id = {v: k for k, v in id_to_piece.items()}

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_piece[i] for i in ids)

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in text:
            if ch not in self.piece_to_id:
                raise KeyError(ch)
            out.append(self.piece_to_id[ch])
        return out


class ScriptedModel(torch.nn.Module):
    """Deterministic toy model whose next token depends on decode step index."""

    def __init__(self, vocab_size: int, schedule: dict[int, int], default_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.schedule = dict(schedule)
        self.default_id = int(default_id)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def _pick(self, abs_pos: int) -> int:
        return int(self.schedule.get(abs_pos, self.default_id))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        logits = torch.full((B, T, self.vocab_size), -10.0, device=device)
        past_len = 0
        if past_key_values:
            past_len = int(past_key_values[0][0].shape[2])
        for t in range(T):
            abs_pos = past_len + t
            next_id = self._pick(abs_pos)
            logits[:, t, next_id] = 10.0
        if not use_cache:
            return logits, None
        total = past_len + T
        k = torch.zeros(B, 1, total, 1, device=device)
        v = torch.zeros(B, 1, total, 1, device=device)
        return logits, [(k, v)]


def test_eos_stops_generation_excludes_or_includes_eos_by_policy():
    tok = PieceTokenizer({0: "a", 1: "b", 2: "<eos>"})
    model = ScriptedModel(vocab_size=3, schedule={2: 2}, default_id=0)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out = engine.generate(
        prompt_ids=[0, 1],
        attention_mask=[1, 1],
        max_new_tokens=5,
        temperature=0.0,
        eos_token_id=2,
    )
    assert out["stop_reason"] == "eos"
    assert out["completion_token_ids"] == []
    assert out["completion_text"] == ""


def test_stop_token_ids_stops_before_emitting_token_or_after_emitting_token_by_policy():
    tok = PieceTokenizer({0: "x", 1: "y", 2: "!"})
    model = ScriptedModel(vocab_size=3, schedule={1: 2}, default_id=0)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=4,
        temperature=0.0,
        stop_token_ids={2},
    )
    assert out["stop_reason"] == "stop_token"
    assert 2 not in out["completion_token_ids"]


def test_stop_string_stops_on_decoded_boundary_and_handles_overlap():
    tok = PieceTokenizer({0: "a", 1: "#", 2: "##", 3: "b"})
    # prompt len=1 => first decode step abs_pos=1
    model = ScriptedModel(vocab_size=4, schedule={1: 0, 2: 1, 3: 2, 4: 3}, default_id=3)
    engine = Engine(model, tok, block_size=32, max_cache_len=32)

    out = engine.generate(
        prompt_ids=[3],
        attention_mask=[1],
        max_new_tokens=5,
        temperature=0.0,
        stop_strings=["###"],
    )
    assert out["stop_reason"] == "stop_string"
    assert out["completion_text"] == "a"
    assert out["completion_token_ids"] == [0]


def test_stop_string_multiple_occurrences_trims_at_first_match():
    tok = PieceTokenizer({0: "p", 1: "q", 2: "abab"})
    # prompt len=1 and scripted fixture bootstrap means first generated comes from abs_pos=1.
    model = ScriptedModel(vocab_size=3, schedule={1: 2}, default_id=1)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=4,
        temperature=0.0,
        stop_strings=["ab"],
    )
    assert out["stop_reason"] == "stop_string"
    # Token decodes to "abab"; earliest stop-string occurrence is index 0.
    assert out["completion_text"] == ""
    assert out["completion_token_ids"] == []


def test_token_level_stop_precedence_over_stop_string():
    tok = PieceTokenizer({0: "x", 1: "y", 2: "STOP"})
    model = ScriptedModel(vocab_size=3, schedule={1: 2}, default_id=1)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=4,
        temperature=0.0,
        stop_token_ids={2},
        stop_strings=["STOP"],
    )
    # Token-level stops are evaluated before string-level checks.
    assert out["stop_reason"] == "stop_token"
    assert out["completion_token_ids"] == []


def test_max_new_tokens_halts_exactly():
    tok = PieceTokenizer({0: "a", 1: "b"})
    model = ScriptedModel(vocab_size=2, schedule={}, default_id=1)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=3,
        temperature=0.0,
    )
    assert out["stop_reason"] == "max_new_tokens"
    assert len(out["completion_token_ids"]) == 3


def test_context_truncation_keeps_last_block_size_tokens_and_cache_truncates():
    p, g = apply_context_truncation([1, 2, 3, 4], [5, 6, 7], block_size=5)
    assert len(p) + len(g) <= 5
    assert p + g == [3, 4, 5, 6, 7]

    k = torch.zeros(1, 1, 9, 1)
    v = torch.zeros(1, 1, 9, 1)
    state = CacheState(past_key_values=[(k, v)], past_len=9)
    truncated = truncate_kv_cache_to_block_size(state, keep_last_n=4)
    assert truncated.past_len == 4
    assert int(truncated.past_key_values[0][0].shape[2]) == 4


def test_truncation_is_consistent_with_cache_vs_recompute(serving_pkg, serving_device: str):
    cfg, tok, model = serving_pkg
    model.eval()
    torch.manual_seed(0)

    block_size = int(getattr(cfg, "block_size", 128))
    vocab = int(getattr(cfg, "vocab_size", 10000))
    prompt = torch.randint(0, vocab, (1, min(block_size + 8, 32)), dtype=torch.long, device=serving_device)
    prompt_ids = prompt[0].tolist()

    engine = Engine(model, tok, block_size=block_size, max_cache_len=block_size)
    out = engine.generate(
        prompt_ids=prompt_ids,
        attention_mask=[1] * len(prompt_ids),
        max_new_tokens=5,
        temperature=0.0,
    )

    recompute_ids = list(prompt_ids)
    expected = []
    for _ in range(5):
        cur = recompute_ids[-block_size:]
        x = torch.tensor([cur], dtype=torch.long, device=serving_device)
        logits, _ = model(x, attention_mask=torch.ones_like(x), past_key_values=None, use_cache=False)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        expected.append(next_id)
        recompute_ids.append(next_id)
    assert out["completion_token_ids"] == expected


def test_invalid_inputs_fail_fast():
    tok = PieceTokenizer({0: "x"})
    model = ScriptedModel(vocab_size=1, schedule={})
    engine = Engine(model, tok, block_size=8, max_cache_len=8)

    try:
        _ = engine.generate(prompt_ids=[0], attention_mask=[1], max_new_tokens=-1)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        _ = apply_context_truncation([0], [0], block_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    stopped, reason = should_stop_text("abc", stop_strings=["zzz"])
    assert stopped is False and reason is None
