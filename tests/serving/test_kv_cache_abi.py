# tests/serving/test_kv_cache_abi.py
from __future__ import annotations

import torch


def _shape_str(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} contig={x.is_contiguous()}"


def test_past_kv_shapes_and_growth(serving_pkg, serving_device: str):
    """
    Invariant:
      - past is list of length n_layers
      - each entry is (k, v) tensors
      - time dimension grows by +1 when decoding with T_new=1
    """
    config, tok, model = serving_pkg
    device = serving_device

    vocab = int(getattr(config, "vocab_size", 10000))
    n_layers = int(getattr(config, "n_layers", 1))

    B, T = 2, 16
    x = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
    m = torch.ones((B, T), dtype=torch.long, device=device)

    with torch.no_grad():
        logits0, past0 = model(x, attention_mask=m, past_key_values=None, use_cache=True)

    assert past0 is not None
    assert isinstance(past0, list)
    assert len(past0) == n_layers

    k0, v0 = past0[0]
    assert isinstance(k0, torch.Tensor) and isinstance(v0, torch.Tensor)
    assert k0.dtype == v0.dtype
    assert k0.device == v0.device

    # Heuristic: time dim is one of the dims; we require that some dim equals T.
    assert T in k0.shape, f"Expected time dim == T somewhere; got {_shape_str(k0)}"

    # Decode step with a single new token
    x1 = torch.randint(0, vocab, (B, 1), dtype=torch.long, device=device)
    m1 = torch.ones((B, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        logits1, past1 = model(x1, attention_mask=m1, past_key_values=past0, use_cache=True)

    assert past1 is not None
    assert len(past1) == n_layers
    k1, v1 = past1[0]
    assert k1.shape == v1.shape

    # Require growth by +1 in exactly one dimension compared to k0
    diffs = [b - a for a, b in zip(k0.shape, k1.shape)]
    assert diffs.count(1) == 1, f"Expected exactly one dim to grow by +1. k0={k0.shape} k1={k1.shape} diffs={diffs}"
    assert diffs.count(0) == len(diffs) - 1, f"Unexpected multi-dim changes. k0={k0.shape} k1={k1.shape}"

    # Logits shapes sanity
    assert logits0.shape[:2] == (B, T)
    assert logits1.shape[:2] == (B, 1)