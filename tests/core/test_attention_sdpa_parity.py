# tests/core/test_attention_sdpa_parity.py
"""
Numerical parity: SDPA path must produce outputs close to the raw attention path.
Uses float32 throughout to avoid bf16 accumulation differences.
Same weights, same inputs — only the attention-compute flag differs.
"""
import torch
import pytest
from llm_lab.core.model.attention import (
    SingleHeadAttention, SingleHeadAttentionConfig,
    MultiHeadAttention, MultiHeadAttentionConfig,
)

_ATOL = 1e-5  # float32 tolerance; SDPA and manual softmax are algebraically identical


def _make_sha(use_sdpa: bool, **kwargs) -> SingleHeadAttention:
    defaults = dict(d_model=32, head_dim=16, dropout=0.0, use_rope=False)
    defaults.update(kwargs)
    return SingleHeadAttention(SingleHeadAttentionConfig(**defaults, use_sdpa=use_sdpa)).eval()


def _make_gqa(use_sdpa: bool, **kwargs) -> MultiHeadAttention:
    defaults = dict(d_model=64, n_heads=8, dropout=0.0, attention_type="gqa",
                    num_kv_heads=2, use_rope=False)
    defaults.update(kwargs)
    return MultiHeadAttention(MultiHeadAttentionConfig(**defaults, use_sdpa=use_sdpa)).eval()


# ── SingleHeadAttention parity ────────────────────────────────────────────────

def test_sha_parity_no_mask():
    torch.manual_seed(42)
    x = torch.randn(2, 10, 32)
    raw  = _make_sha(use_sdpa=False)
    sdpa = _make_sha(use_sdpa=True)
    sdpa.load_state_dict(raw.state_dict())
    with torch.no_grad():
        y_raw,  _ = raw(x)
        y_sdpa, _ = sdpa(x)
    torch.testing.assert_close(y_raw, y_sdpa, atol=_ATOL, rtol=0.0)


def test_sha_parity_with_padding_mask():
    torch.manual_seed(42)
    B, T = 2, 10
    x = torch.randn(B, T, 32)
    mask = torch.ones(B, T, dtype=torch.int64)
    mask[:, -2:] = 0  # last two tokens are pad
    raw  = _make_sha(use_sdpa=False)
    sdpa = _make_sha(use_sdpa=True)
    sdpa.load_state_dict(raw.state_dict())
    with torch.no_grad():
        y_raw,  _ = raw(x,  attention_mask=mask)
        y_sdpa, _ = sdpa(x, attention_mask=mask)
    torch.testing.assert_close(y_raw, y_sdpa, atol=_ATOL, rtol=0.0)


# ── GQA parity ────────────────────────────────────────────────────────────────

def test_gqa_parity_no_mask():
    torch.manual_seed(42)
    x = torch.randn(2, 10, 64)
    raw  = _make_gqa(use_sdpa=False)
    sdpa = _make_gqa(use_sdpa=True)
    sdpa.load_state_dict(raw.state_dict())
    with torch.no_grad():
        y_raw,  _ = raw(x)
        y_sdpa, _ = sdpa(x)
    torch.testing.assert_close(y_raw, y_sdpa, atol=_ATOL, rtol=0.0)


def test_gqa_parity_with_padding_mask():
    torch.manual_seed(42)
    B, T = 2, 10
    x = torch.randn(B, T, 64)
    mask = torch.ones(B, T, dtype=torch.int64)
    mask[:, -2:] = 0
    raw  = _make_gqa(use_sdpa=False)
    sdpa = _make_gqa(use_sdpa=True)
    sdpa.load_state_dict(raw.state_dict())
    with torch.no_grad():
        y_raw,  _ = raw(x,  attention_mask=mask)
        y_sdpa, _ = sdpa(x, attention_mask=mask)
    torch.testing.assert_close(y_raw, y_sdpa, atol=_ATOL, rtol=0.0)


def test_gqa_parity_varied_shapes():
    """Different batch sizes, sequence lengths, and head configs all match."""
    configs = [
        dict(d_model=32, n_heads=4, num_kv_heads=1),   # 4:1 GQA
        dict(d_model=64, n_heads=8, num_kv_heads=4),   # 2:1 GQA
        dict(d_model=64, n_heads=8, num_kv_heads=8),   # MQA-style (repeat=1)
    ]
    for cfg_kwargs in configs:
        torch.manual_seed(7)
        B, T = 3, 12
        x = torch.randn(B, T, cfg_kwargs["d_model"])
        raw  = _make_gqa(use_sdpa=False, **cfg_kwargs)
        sdpa = _make_gqa(use_sdpa=True,  **cfg_kwargs)
        sdpa.load_state_dict(raw.state_dict())
        with torch.no_grad():
            y_raw,  _ = raw(x)
            y_sdpa, _ = sdpa(x)
        torch.testing.assert_close(
            y_raw, y_sdpa, atol=_ATOL, rtol=0.0,
            msg=f"parity failed for config {cfg_kwargs}",
        )
