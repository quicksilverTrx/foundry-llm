# tests/serving/test_rope_offset_progression.py
from __future__ import annotations

import torch


def test_rope_offset_cached_matches_recompute_last_step(serving_pkg, serving_device: str):
    """
    Invariant: cached decode logits for the new token match recompute logits
    for the same full context (up to tolerance).
    """
    config, tok, model = serving_pkg
    device = serving_device

    vocab = int(getattr(config, "vocab_size", 10000))
    B, T = 1, 24

    x = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
    m = torch.ones((B, T), dtype=torch.long, device=device)

    with torch.no_grad():
        # Prefill with cache
        logits_prefill, past = model(x, attention_mask=m, past_key_values=None, use_cache=True)

        # Next token
        x_new = torch.randint(0, vocab, (B, 1), dtype=torch.long, device=device)
        m_new = torch.ones((B, 1), dtype=torch.long, device=device)

        logits_cached, _ = model(x_new, attention_mask=m_new, past_key_values=past, use_cache=True)

        # Recompute on full context
        x_full = torch.cat([x, x_new], dim=1)
        m_full = torch.ones_like(x_full)
        logits_full, _ = model(x_full, attention_mask=m_full, past_key_values=None, use_cache=False)

    # Compare last-step logits
    assert torch.allclose(
        logits_cached[:, -1, :],
        logits_full[:, -1, :],
        rtol=1e-4,
        atol=1e-4,
    )