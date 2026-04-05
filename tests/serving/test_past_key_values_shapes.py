# tests/serving/test_past_key_values_shapes.py
from __future__ import annotations

import torch


def _get_time_dim(k: torch.Tensor) -> int:
    """Canonically the time axis sits at index 2 ([B, heads, T, head_dim])."""
    return 2


def test_past_key_values_shapes_prefill_and_decode(serving_pkg, serving_device: str):
    """KV-cache ABI gate: prefill returns cache and decode grows cache by +1 time step."""
    config, tok, model = serving_pkg
    device = serving_device

    model.eval()
    torch.manual_seed(0)

    B, T = 2, 8
    vocab = int(getattr(config, "vocab_size", 10000))

    x = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
    attn = torch.ones((B, T), dtype=torch.long, device=device)

    with torch.no_grad():
        logits, past = model(x, attention_mask=attn, past_key_values=None, use_cache=True)

    assert logits.shape[:2] == (B, T)
    assert past is not None
    assert isinstance(past, list)

    n_layers = int(getattr(config, "n_layers", len(past)))
    assert len(past) == n_layers

    for layer_index, (k0, v0) in enumerate(past):
        assert isinstance(k0, torch.Tensor) and isinstance(v0, torch.Tensor), f"layer {layer_index} must return tensors"
        assert k0.dtype == v0.dtype
        assert k0.device == v0.device
        assert k0.shape == v0.shape
        assert k0.dim() == 4
        assert k0.shape[0] == B
        assert k0.shape[_get_time_dim(k0)] == T

    x1 = torch.randint(0, vocab, (B, 1), dtype=torch.long, device=device)
    attn1 = torch.ones((B, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        logits2, past2 = model(x1, attention_mask=attn1, past_key_values=past, use_cache=True)

    assert logits2.shape[:2] == (B, 1)
    assert past2 is not None
    assert len(past2) == n_layers

    for layer_index, (k1, v1) in enumerate(past2):
        assert k1.shape == v1.shape
        assert k1.shape[_get_time_dim(k1)] == T + 1, f"layer {layer_index} did not grow by +1 on time axis"
