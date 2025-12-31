import torch
from llm_lab.core.model.pos_encodings import apply_rope

def test_rope_is_identity_at_position_zero():
    torch.manual_seed(0)
    B, T, H = 2, 5, 8  # head_dim must be even
    q = torch.randn(B, T, H)
    k = torch.randn(B, T, H)
    pos = torch.zeros(T, dtype=torch.long)

    q2, k2 = apply_rope(q, k, pos)

    assert torch.allclose(q2, q, atol=1e-6, rtol=1e-6)
    assert torch.allclose(k2, k, atol=1e-6, rtol=1e-6)
