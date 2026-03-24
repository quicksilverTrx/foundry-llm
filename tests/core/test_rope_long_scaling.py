# tests/core/test_rope_long_scaling.py
import torch

def test_rope_linear_factor_1_matches_none_close():
    from llm_lab.core.model.pos_encodings import apply_rope

    torch.manual_seed(0)
    B, T, D = 2, 64, 32
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    pos = torch.arange(T)

    q0, k0 = apply_rope(q, k, pos, rope_scaling_type="none", rope_scaling_factor=1.0, position_offset=0)
    q1, k1 = apply_rope(q, k, pos, rope_scaling_type="linear", rope_scaling_factor=1.0, position_offset=0)

    assert torch.allclose(q0, q1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k0, k1, atol=1e-5, rtol=1e-5)


def test_rope_linear_scaling_stays_finite_long_seq():
    from llm_lab.core.model.pos_encodings import apply_rope

    torch.manual_seed(0)
    B, T, D = 1, 1024, 64
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    pos = torch.arange(T)

    q2, k2 = apply_rope(q, k, pos, rope_scaling_type="linear", rope_scaling_factor=4.0, position_offset=0)
    assert torch.isfinite(q2).all()
    assert torch.isfinite(k2).all()


def test_rope_position_offset_changes_output():
    from llm_lab.core.model.pos_encodings import apply_rope

    torch.manual_seed(0)
    B, T, D = 1, 128, 64
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    pos = torch.arange(T)

    q0, k0 = apply_rope(q, k, pos, position_offset=0)
    q1, k1 = apply_rope(q, k, pos, position_offset=1000)

    assert torch.isfinite(q1).all()
    assert not torch.allclose(q0, q1)
