#test_attention_gqa.pyimport torch
import pytest
import torch
def test_gqa_shapes_and_param_savings_vs_mha():
    torch.manual_seed(0)
    B, T, d_model = 2, 8, 64
    n_heads = 8
    num_kv_heads = 2

    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig

    gqa = MultiHeadAttention(MultiHeadAttentionConfig(
        d_model=d_model, n_heads=n_heads, dropout=0.0,
        attention_type="gqa", num_kv_heads=num_kv_heads,
        use_rope=False,
    ))
    x = torch.randn(B, T, d_model)
    y, _ = gqa(x)
    assert y.shape == (B, T, d_model)

    mha = MultiHeadAttention(MultiHeadAttentionConfig(
        d_model=d_model, n_heads=n_heads, dropout=0.0,
        attention_type="mha",
        use_rope=False,
    ))

    mha_k = sum(h.k_proj.weight.numel() for h in mha.heads)
    mha_v = sum(h.v_proj.weight.numel() for h in mha.heads)

    assert gqa.k_proj.weight.numel() < mha_k
    assert gqa.v_proj.weight.numel() < mha_v

def test_gqa_causality_no_future_leak():
    torch.manual_seed(0)
    B, T, d_model = 1, 6, 64
    n_heads, num_kv_heads = 8, 2

    from llm_lab.core.model.attention import MultiHeadAttention,MultiHeadAttentionConfig
    attn = MultiHeadAttention(
        MultiHeadAttentionConfig(d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
        attention_type="gqa",
        num_kv_heads=num_kv_heads,)
    )

    x = torch.randn(B, T, d_model)
    y1,_ = attn(x)
    y1 = y1.detach()

    # perturb FUTURE token strongly
    x2 = x.clone()
    x2[:, -1, :] += 1000.0
    y2,_ = attn(x2)
    y2 = y2.detach()

    # earlier positions must not change
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5, rtol=1e-5)

def _set_linear_identity(linear: torch.nn.Linear):
    with torch.no_grad():
        linear.weight.zero_()
        n = min(linear.weight.shape[0], linear.weight.shape[1])
        linear.weight[:n, :n] = torch.eye(n)
        if linear.bias is not None:
            linear.bias.zero_()

def test_gqa_concat_order_regression_identity_like():
    torch.manual_seed(0)
    B, T = 1, 4
    H, D = 2, 4
    d_model = H * D
    alpha = 50.0

    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    attn = MultiHeadAttention(MultiHeadAttentionConfig(
        d_model=d_model, n_heads=H, dropout=0.0,
        attention_type="gqa", num_kv_heads=H,  # repeat=1, still uses your GQA path
        use_rope=False,
    ))

    # Make projections identity so q,k,v ~= x (after reshape/transpose)
    _set_linear_identity(attn.q_proj)
    _set_linear_identity(attn.k_proj)
    _set_linear_identity(attn.v_proj)
    _set_linear_identity(attn.out_proj)

    # Build x so that within EACH head, token t has an orthogonal basis vector e_t
    # head0 dims: [0..3], head1 dims: [4..7]
    x = torch.zeros(B, T, d_model)
    for t in range(T):
        x[0, t, t] = alpha          # head 0, dim t
        x[0, t, D + t] = alpha      # head 1, dim t

    y, _ = attn(x)

    # Because q·k is huge on the diagonal and ~0 off-diagonal, attention is ~identity.
    # If you forget transpose-before-flatten, this will FAIL hard.
    assert torch.allclose(y, x, atol=1e-2, rtol=0.0)

import pytest

def test_gqa_invalid_num_kv_heads_raises():
    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    with pytest.raises((AssertionError, ValueError)):
        MultiHeadAttention(MultiHeadAttentionConfig(
            d_model=64, n_heads=8, dropout=0.0,
            attention_type="gqa", num_kv_heads=3,
        ))
def test_attention_mask_blocks_keys():
    torch.manual_seed(0)
    B, T, d_model = 1, 6, 64
    n_heads, num_kv_heads = 8, 2

    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    attn = MultiHeadAttention(MultiHeadAttentionConfig(
        d_model=d_model, n_heads=n_heads, dropout=0.0,
        attention_type="gqa", num_kv_heads=num_kv_heads,
        use_rope=False,
    ))

    x = torch.randn(B, T, d_model)
    mask = torch.ones(B, T, dtype=torch.int64)
    mask[:, -1] = 0  # last token is "pad", should not be attended to as a key

    y1, _ = attn(x, attention_mask=mask)
    x2 = x.clone()
    x2[:, -1, :] += 1000.0  # perturb masked token
    y2, _ = attn(x2, attention_mask=mask)

    # if key is masked, earlier outputs shouldn't change
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5, rtol=1e-5)

import pytest
def test_attention_mask_shape_validation():
    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    attn = MultiHeadAttention(MultiHeadAttentionConfig(
        d_model=64, n_heads=8, dropout=0.0, attention_type="gqa", num_kv_heads=2
    ))
    x = torch.randn(1, 6, 64)
    with pytest.raises(ValueError):
        attn(x, attention_mask=torch.ones(1, 1, 6))  # wrong rank
