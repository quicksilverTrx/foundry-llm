# tests/serving/test_attention_mask_padding.py
from __future__ import annotations

import torch


def _right_pad(ids: list[int], T: int, pad_id: int) -> list[int]:
    assert len(ids) <= T
    return ids + [pad_id] * (T - len(ids))


def test_padding_equivalence_logits_match_nonpad_positions(serving_pkg, serving_device: str):
    """
    Invariant: batched right-padded prefill must match unpadded single-run logits
    on the non-pad query positions.
    """
    config, tok, model = serving_pkg
    device = serving_device

    # --- Build two different-length sequences ---
    # Keep it deterministic and small.
    text_a = "To be, or not to be"
    text_b = "To be"

    ids_a = tok.encode(text_a)
    ids_b = tok.encode(text_b)

    # If tokenizer has special tokens, we don't need them here.
    pad_id = getattr(tok, "pad_token_id", 0)

    T = max(len(ids_a), len(ids_b))
    a = torch.tensor(ids_a, dtype=torch.long, device=device)[None, :]
    b = torch.tensor(_right_pad(ids_b, T, pad_id), dtype=torch.long, device=device)[None, :]

    # Pad A too (in case it's shorter for some tokenizer edge case)
    if a.shape[1] != T:
        a = torch.tensor(_right_pad(ids_a, T, pad_id), dtype=torch.long, device=device)[None, :]

    input_ids = torch.cat([a, b], dim=0)  # [2,T]

    attn_mask = torch.zeros((2, T), dtype=torch.long, device=device)
    attn_mask[0, : len(ids_a)] = 1
    attn_mask[1, : len(ids_b)] = 1

    with torch.no_grad():
        logits_batched, _ = model(input_ids, attention_mask=attn_mask, use_cache=False)

        b_unpadded = torch.tensor(ids_b, dtype=torch.long, device=device)[None, :]
        m_unpadded = torch.ones((1, len(ids_b)), dtype=torch.long, device=device)
        logits_single, _ = model(b_unpadded, attention_mask=m_unpadded, use_cache=False)

    Lb = len(ids_b)
    assert logits_batched.shape[0] == 2
    assert logits_batched.shape[1] == T
    assert logits_single.shape[1] == Lb

    # Compare only B's non-pad query positions
    assert torch.allclose(
        logits_batched[1, :Lb, :],
        logits_single[0, :, :],
        rtol=1e-4,
        atol=1e-4,
    )


def test_all_pad_rows_do_not_nan(serving_pkg, serving_device: str):
    """
    Regression: extreme padding should not produce NaNs.
    """
    config, tok, model = serving_pkg
    device = serving_device

    pad_id = getattr(tok, "pad_token_id", 0)
    T = min(int(getattr(config, "block_size", 64)), 64)

    input_ids = torch.full((2, T), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((2, T), dtype=torch.long, device=device)  # all pad

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask=attn_mask, use_cache=False)

    assert torch.isfinite(logits).all()