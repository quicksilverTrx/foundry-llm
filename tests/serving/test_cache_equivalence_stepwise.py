# tests/serving/test_cache_equivalence_stepwise.py
from __future__ import annotations

import torch

from llm_lab.serving.engine import Engine


def greedy_next(logits_last: torch.Tensor) -> int:
    return int(logits_last.argmax(dim=-1).item())


def run_recompute(model, prompt_ids: torch.Tensor, steps: int) -> list[torch.Tensor]:
    model.eval()
    output_ids = prompt_ids
    output_logits = []
    for _ in range(steps):
        # Recompute path always runs on full current context without cache.
        logits, _ = model(output_ids, attention_mask=torch.ones_like(output_ids), use_cache=False)
        logits_last = logits[:, -1, :]
        output_logits.append(logits_last)
        next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
    return output_logits


def run_cached(engine: Engine, prompt_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[torch.Tensor]:
    # Cached path does exactly one prefill, then token-wise decode steps.
    logits, cache, _ = engine.prefill(prompt_ids, attention_mask)
    output_logits = []
    output_logits.append(logits)
    for _ in range(steps - 1):
        input_id = torch.argmax(logits, dim=-1, keepdim=True)
        logits, cache = engine.decode_step(input_id, cache)
        output_logits.append(logits)
    return output_logits


def test_equivalence_scaffold_is_deterministic(serving_pkg, serving_device: str):
    config, tok, model = serving_pkg
    model.eval()
    torch.manual_seed(0)

    vocab = int(getattr(config, "vocab_size", 10000))
    prompt = torch.randint(0, vocab, (1, 8), dtype=torch.long, device=serving_device)
    mask = torch.ones_like(prompt)

    engine = Engine(model, tok, block_size=int(getattr(config, "block_size", 128)))

    recompute_logits = run_recompute(model, prompt, steps=3)
    cached_logits = run_cached(engine, prompt, mask, steps=3)
    for x, y in zip(cached_logits, recompute_logits):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-4)
