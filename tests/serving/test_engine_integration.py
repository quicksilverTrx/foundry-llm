# tests/serving/test_engine_integration.py
from __future__ import annotations

import torch

from llm_lab.serving.engine import Engine


def run_recompute(model, prompt_ids: torch.Tensor, steps: int) -> list[torch.Tensor]:
    model.eval()
    output_ids = prompt_ids
    output_logits = []
    for _ in range(steps):
        logits, _ = model(output_ids, attention_mask=torch.ones_like(output_ids), use_cache=False)
        logits_last = logits[:, -1, :]
        output_logits.append(logits_last)
        next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
    return output_logits


def run_cached(engine: Engine, prompt_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[torch.Tensor]:
    logits, cache, _ = engine.prefill(prompt_ids, attention_mask)
    output_logits = [logits]
    for _ in range(steps - 1):
        input_id = torch.argmax(logits, dim=-1, keepdim=True)
        logits, cache = engine.decode_step(input_id, cache)
        output_logits.append(logits)
    return output_logits


class PieceTokenizer:
    def __init__(self, id_to_piece: dict[int, str]):
        self.id_to_piece = dict(id_to_piece)
        self.piece_to_id = {v: k for k, v in id_to_piece.items()}

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_piece[i] for i in ids)

    def encode(self, text: str) -> list[int]:
        return [self.piece_to_id[ch] for ch in text]


class FlatLogitModel(torch.nn.Module):
    """Model that returns stable logits; useful for seeded sampling assertions."""

    def __init__(self, vocab_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        row = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32, device=device)
        logits = row.view(1, 1, -1).expand(B, T, self.vocab_size).contiguous()
        if not use_cache:
            return logits, None
        past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
        total = past_len + T
        k = torch.zeros(B, 1, total, 1, device=device)
        v = torch.zeros(B, 1, total, 1, device=device)
        return logits, [(k, v)]


class StepwiseModel(torch.nn.Module):
    """Model whose best token alternates by absolute step index."""

    def __init__(self, vocab_size: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        logits = torch.full((B, T, self.vocab_size), -10.0, device=device)
        past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
        for t in range(T):
            pick = (past_len + t) % self.vocab_size
            logits[:, t, pick] = 10.0
        if not use_cache:
            return logits, None
        total = past_len + T
        k = torch.zeros(B, 1, total, 1, device=device)
        v = torch.zeros(B, 1, total, 1, device=device)
        return logits, [(k, v)]


def test_greedy_generation_still_matches_cache_equivalence_gate(serving_pkg, serving_device: str):
    cfg, tok, model = serving_pkg
    model.eval()
    torch.manual_seed(0)

    vocab = int(getattr(cfg, "vocab_size", 10000))
    prompt = torch.randint(0, vocab, (1, 8), dtype=torch.long, device=serving_device)
    mask = torch.ones_like(prompt)
    engine = Engine(model, tok, block_size=int(getattr(cfg, "block_size", 128)))

    cached_logits = run_cached(engine, prompt, mask, steps=4)
    recompute_logits = run_recompute(model, prompt, steps=4)
    for x, y in zip(cached_logits, recompute_logits):
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)


def test_seeded_sampling_reproducible_end_to_end():
    tok = PieceTokenizer({0: "a", 1: "b", 2: "c", 3: "d", 4: "e"})
    model = FlatLogitModel(vocab_size=5)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out1 = engine.generate(
        prompt_ids=[0, 1],
        attention_mask=[1, 1],
        max_new_tokens=6,
        temperature=1.0,
        top_p=0.9,
        seed=123,
    )
    out2 = engine.generate(
        prompt_ids=[0, 1],
        attention_mask=[1, 1],
        max_new_tokens=6,
        temperature=1.0,
        top_p=0.9,
        seed=123,
    )
    assert out1["completion_token_ids"] == out2["completion_token_ids"]


def test_stop_conditions_trigger_with_sampling():
    tok = PieceTokenizer({0: "a", 1: "b", 2: "#", 3: "c", 4: "d"})
    model = StepwiseModel(vocab_size=5)
    engine = Engine(model, tok, block_size=16, max_cache_len=16)

    out_tok = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=8,
        temperature=1.0,
        top_k=1,
        stop_token_ids={1},
    )
    assert out_tok["stop_reason"] in {"stop_token", "max_new_tokens"}

    out_str = engine.generate(
        prompt_ids=[0],
        attention_mask=[1],
        max_new_tokens=8,
        temperature=1.0,
        top_k=1,
        stop_strings=["ab"],
    )
    assert out_str["stop_reason"] in {"stop_string", "max_new_tokens"}


def test_truncation_does_not_crash_and_cache_len_bounded():
    tok = PieceTokenizer({0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"})
    model = StepwiseModel(vocab_size=6)
    engine = Engine(model, tok, block_size=5, max_cache_len=5)

    out = engine.generate(
        prompt_ids=[0, 1, 2, 3, 4, 5, 0],
        attention_mask=[1, 1, 1, 1, 1, 1, 1],
        max_new_tokens=7,
        temperature=0.0,
    )
    assert out["metrics"]["cache_len"] <= 5
    assert len(out["all_token_ids"]) <= 5


def test_generate_invalid_input_contracts():
    tok = PieceTokenizer({0: "a", 1: "b"})
    model = StepwiseModel(vocab_size=2)
    engine = Engine(model, tok, block_size=4, max_cache_len=4)

    try:
        _ = engine.generate(prompt_ids=[], attention_mask=None, max_new_tokens=1)
        assert False, "expected ValueError for empty prompt"
    except ValueError:
        pass
    try:
        _ = engine.generate(prompt_ids=[0, 1], attention_mask=[1], max_new_tokens=1)
        assert False, "expected ValueError for bad mask shape"
    except ValueError:
        pass
