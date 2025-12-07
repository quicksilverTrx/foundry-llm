import torch

from llm_lab.core.model import MultiHeadAttentionConfig,MultiHeadAttention
def test_attention():
    cfg = MultiHeadAttentionConfig(d_model=32, n_heads=4, dropout=0.0)
    mha = MultiHeadAttention(cfg)
    x = torch.randn(2, 5, 32)
    y = mha(x)
    assert y.shape == (2, 5, 32)