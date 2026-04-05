# tests/serving/test_package_smoke.py
from __future__ import annotations

import torch


def test_package_smoke_forward_no_cache(serving_pkg, serving_device: str):
    """Smoke gate: package loads and basic forward pass is numerically stable."""
    config, tok, model = serving_pkg
    device = serving_device

    model.eval()
    torch.manual_seed(0)

    B, T = 1, 4
    vocab = int(getattr(config, "vocab_size", 10000))

    x = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, T), dtype=torch.long, device=device)

    with torch.no_grad():
        logits, past = model(x, attention_mask=attention_mask, use_cache=False)

    assert logits.shape[:2] == (B, T)
    assert logits.shape[-1] == vocab
    assert torch.isfinite(logits).all()
    assert past is None
