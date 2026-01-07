import torch
import pytest

import llm_lab.core.model.attention as attn_mod
from llm_lab.core.model.attention import (
    SingleHeadAttention, SingleHeadAttentionConfig,
    MultiHeadAttention, MultiHeadAttentionConfig,
)

def _clear_mask_cache():
    attn_mod._CAUSAL_MASK_CACHE.clear()

def _set_use_cache(flag: bool):
    # global switch in your module
    attn_mod.USE_CACHE = flag

@pytest.mark.parametrize("T", [1, 2, 7, 16])
def test_causal_mask_values(T):
    _clear_mask_cache()
    _set_use_cache(True)

    # NOTE: _causal_mask is now module-level (not on SingleHeadAttention)
    mask = attn_mod._causal_mask(T, device=torch.device("cpu"), dtype=torch.float32)

    assert mask.shape == (T, T)

    # Allowed positions (j <= i) should be 0
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)
    assert torch.all(mask[lower] == 0.0)

    # Future positions (j > i) should be very negative
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(mask[upper] <= -1e8)

def test_mask_cache_keyed_by_T_device_dtype():
    _clear_mask_cache()
    _set_use_cache(True)

    # same call should reuse
    _ = attn_mod._causal_mask(8, torch.device("cpu"), torch.float32)
    n1 = len(attn_mod._CAUSAL_MASK_CACHE)

    _ = attn_mod._causal_mask(8, torch.device("cpu"), torch.float32)
    n2 = len(attn_mod._CAUSAL_MASK_CACHE)
    assert n2 == n1

    # different T should add
    _ = attn_mod._causal_mask(16, torch.device("cpu"), torch.float32)
    assert len(attn_mod._CAUSAL_MASK_CACHE) == n1 + 1

    # different dtype should add (float16 mask creation is fine on CPU)
    _ = attn_mod._causal_mask(8, torch.device("cpu"), torch.float16)
    assert len(attn_mod._CAUSAL_MASK_CACHE) == n1 + 2

def test_single_head_cache_on_off_equivalence_eval():
    """
    The mask cache must not change outputs.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)  # [B,T,d_model]

    cfg = SingleHeadAttentionConfig(d_model=16, head_dim=8, dropout=0.0, use_rope=False)
    head = SingleHeadAttention(cfg).eval()

    _clear_mask_cache()
    _set_use_cache(False)
    with torch.no_grad():
        y_off, _ = head(x)

    _clear_mask_cache()
    _set_use_cache(True)
    with torch.no_grad():
        y_on, _ = head(x)

    assert torch.isfinite(y_off).all()
    assert torch.isfinite(y_on).all()
    torch.testing.assert_close(y_off, y_on, rtol=0.0, atol=0.0)  # should be exact

def test_multi_head_cache_on_off_equivalence_eval():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32)

    cfg = MultiHeadAttentionConfig(d_model=32, n_heads=4, dropout=0.0, use_rope=False)
    mha = MultiHeadAttention(cfg).eval()

    _clear_mask_cache()
    _set_use_cache(False)
    with torch.no_grad():
        y_off, _ = mha(x)

    _clear_mask_cache()
    _set_use_cache(True)
    with torch.no_grad():
        y_on, _ = mha(x)

    assert torch.isfinite(y_off).all()
    assert torch.isfinite(y_on).all()
    torch.testing.assert_close(y_off, y_on, rtol=0.0, atol=0.0)

def test_no_future_leak_single_head():
    """
    Change x at a future position; earlier outputs must not change.
    """
    torch.manual_seed(0)
    B, T, C = 2, 12, 16
    t_future = 9

    cfg = SingleHeadAttentionConfig(d_model=C, head_dim=8, dropout=0.0, use_rope=False)
    head = SingleHeadAttention(cfg).eval()

    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[:, t_future, :] += 1000.0  # huge perturbation at a future token

    _clear_mask_cache()
    _set_use_cache(True)
    with torch.no_grad():
        y1, _ = head(x1)
        y2, _ = head(x2)

    # Earlier positions (< t_future) must match exactly
    torch.testing.assert_close(y1[:, :t_future, :], y2[:, :t_future, :], rtol=0.0, atol=0.0)
