# tests/serving/test_rope_position_offset_progression.py
from __future__ import annotations

import torch


def test_rope_offset_cached_matches_recompute_last_step(serving_pkg, serving_device: str):
    """RoPE correctness gate: cached decode logits must match full recompute last step."""
    config, tok, model = serving_pkg
    device = serving_device

    model.eval()
    torch.manual_seed(0)

    B, T = 1, 12
    vocab = int(getattr(config, "vocab_size", 10000))

    prompt = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
    attn = torch.ones((B, T), dtype=torch.long, device=device)

    with torch.no_grad():
        logits_prefill, past = model(prompt, attention_mask=attn, past_key_values=None, use_cache=True)

        x_new = torch.randint(0, vocab, (B, 1), dtype=torch.long, device=device)
        attn_new = torch.ones((B, 1), dtype=torch.long, device=device)

        logits_cached, _ = model(
            x_new,
            attention_mask=attn_new,
            past_key_values=past,
            use_cache=True,
        )

        full = torch.cat([prompt, x_new], dim=1)
        logits_full, _ = model(full, attention_mask=torch.ones_like(full), use_cache=False)

    assert torch.allclose(
        logits_cached[:, 0, :],
        logits_full[:, -1, :],
        rtol=1e-4,
        atol=1e-4,
    )
