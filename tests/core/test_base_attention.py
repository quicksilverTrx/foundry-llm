import torch

from llm_lab.core.model import MultiHeadAttentionConfig,MultiHeadAttention
def test_attention():
    cfg = MultiHeadAttentionConfig(d_model=32, n_heads=4, dropout=0.0)
    mha = MultiHeadAttention(cfg)
    x = torch.randn(2, 5, 32)
    y,_ = mha(x)
    assert y.shape == (2, 5, 32)

def test_attention_is_causal():
    cfg = MultiHeadAttentionConfig(d_model=16, n_heads=2, dropout=0.0)
    attn = MultiHeadAttention(cfg)
    torch.manual_seed(0)

    B, T, C = 1, 4, 16
    x = torch.zeros(B, T, C)

    # baseline
    y1,_ = attn(x)
    y1 = y1.detach()
    # change only the last token
    x2 = x.clone()
    x2[:, -1, :] = torch.randn_like(x2[:, -1, :])
    y2, _= attn(x2)
    y2 = y2.detach()
    # earlier positions must not change
    assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-5)
