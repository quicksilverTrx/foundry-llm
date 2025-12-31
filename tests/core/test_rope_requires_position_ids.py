import torch
import pytest
from llm_lab.core.model.attention import SingleHeadAttention, SingleHeadAttentionConfig

def test_rope_requires_position_ids():
    cfg = SingleHeadAttentionConfig(d_model=16, head_dim=8, dropout=0.0, use_rope=True)
    attn = SingleHeadAttention(cfg)
    x = torch.randn(2, 5, 16)

    with pytest.raises(ValueError):
        attn(x, position_ids=None)
